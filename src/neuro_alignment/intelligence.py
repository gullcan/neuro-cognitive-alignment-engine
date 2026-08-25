from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

import structlog
from openai import AsyncOpenAI

from neuro_alignment.config import Settings
from neuro_alignment.domain import (
    BehaviorEvidence,
    DailyPlan,
    DailyPlanItem,
    NeuroFeedback,
    NormalizedInboundEvent,
    NotionTask,
    TaskAction,
)

logger = structlog.get_logger()


class IntelligenceProvider(Protocol):
    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan: ...

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback: ...


class RuleBasedIntelligenceProvider:
    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan:
        commitment_rank = {"Core": 0, "Flexible": 1, "Optional": 2}
        priority_rank = {"P1": 0, "P2": 1, "P3": 2}
        ordered = sorted(
            tasks,
            key=lambda task: (
                commitment_rank.get(task.commitment_tier, 9),
                priority_rank.get(task.priority, 9),
                task.scheduled_start.isoformat() if task.scheduled_start else "99:99",
            ),
        )
        items = [
            DailyPlanItem(
                task_id=task.page_id,
                title=task.title,
                order=index,
                scheduled_start=task.scheduled_start,
                estimated_minutes=task.estimated_minutes,
                commitment_tier=task.commitment_tier,
                priority=task.priority,
                cognitive_load=task.cognitive_load,
                definition_of_done=task.definition_of_done,
                minimum_action=task.minimum_action,
                rationale=(
                    f"{task.commitment_tier} grubu ve {task.priority} önceliğine göre sıralandı."
                ),
            )
            for index, task in enumerate(ordered, start=1)
        ]
        total_minutes = sum(item.estimated_minutes or 0 for item in items)
        warning = None
        if total_minutes > 480:
            warning = (
                f"Planlanan toplam süre {total_minutes} dakika. "
                "Sekiz saat üzerindeki kapasiteyi yeniden değerlendir."
            )
        return DailyPlan(
            plan_date=plan_date,
            headline=(
                f"Bugün için {len(items)} görev var. Hepsini birden düşünmek yerine "
                "sırayla ilerleyeceğiz."
            ),
            items=items,
            capacity_warning=warning,
            generated_by="rule-based-local",
        )

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback:
        action = event.action or TaskAction.BLOCKED
        counts = evidence.counts
        completed = counts.get("task.completed", 0)
        blocked = counts.get("task.blocked", 0)
        skipped = counts.get("task.skipped", 0)
        rescheduled = counts.get("task.rescheduled", 0)

        if evidence.has_sufficient_history:
            pattern = (
                f"Bu görev için kayıtlarda {completed} tamamlama, {blocked} engel, "
                f"{skipped} vazgeçme ve {rescheduled} erteleme olayı var."
            )
            interpretation = (
                "Bu dağılım tekrarlanan davranış bağlamını gösterir; nedeni tek başına kanıtlamaz."
            )
            confidence = 0.78
        else:
            pattern = "Henüz güvenilir bir kişisel örüntü kurmak için yeterli sonuçlanmış olay yok."
            interpretation = (
                "Bu olay başlangıç verisi olarak kullanılacak; sistem neden uydurmayacak."
            )
            confidence = 0.55

        if evidence.similar_episodes:
            similar_counts: dict[str, int] = {}
            for episode in evidence.similar_episodes:
                similar_counts[episode.action.value] = (
                    similar_counts.get(episode.action.value, 0) + 1
                )
            summary = ", ".join(
                f"{count} {action}" for action, count in sorted(similar_counts.items())
            )
            pattern = (
                f"{pattern} Benzer planlama bağlamlarında bulunan geçmiş olaylar: {summary}. "
                "Bu eşleşme bağlamsal yakınlıktır; aynı nedenin kanıtı değildir."
            )

        if action == TaskAction.COMPLETED:
            gap = f"Bunu yaptın: '{task_title}' bitti."
            completion_label = "ilk" if completed <= 1 else f"{completed}."
            intervention = (
                "Bugünkü emeğin, kendine güvenmek için gerçek bir sebep oluşturdu; "
                f"bu, son dönemde tamamladığın {completion_label} planlı görev."
            )
            request = (
                "Bir an durup başlamanı neyin kolaylaştırdığını fark et ve tek cümleyle "
                "yaz; yarın yine işine yarayacak."
            )
        elif action == TaskAction.STARTED:
            gap = f"Başladın; '{task_title}' artık zihnindeki bir iş değil, ilerleyen bir iş."
            intervention = (
                "İlk adımı attın, şimdi ritmi bozma: yalnızca önündeki küçük adıma "
                "12 dakika boyunca odaklan."
            )
            request = "Süre bitince 'Bitirdim' ya da 'Takıldım' diyerek bana haber ver."
        elif action == TaskAction.BLOCKED:
            gap = f"'{task_title}' üzerinde bir yerde takıldın; bu, yolun bittiği anlamına gelmez."
            intervention = (
                "Şimdi kendine dürüstçe sor: ne yapacağını mı bilmiyorsun, birini mi "
                "bekliyorsun, yoksa başlamak mı zor geliyor? Cevabına göre yapabileceğin "
                "en küçük adımı seç."
            )
            request = "Seni durduran şeyi ve şimdi atacağın küçük adımı bir cümleyle yaz."
        elif action == TaskAction.SKIPPED:
            avoidance_count = skipped + rescheduled
            history = (
                f" Bu işi {avoidance_count} kez atladın ya da erteledin. Bu seni başarısız "
                "yapmaz ama aynı döngünün kendiliğinden değişmeyeceğini gösterir."
                if avoidance_count >= 2
                else ""
            )
            gap = f"Bugün şu işi yapmamayı seçtin: '{task_title}'.{history}"
            intervention = (
                "Kendini suçlamak yerine buradan bir şey öğren: seni durduran asıl nedeni "
                "adlandır, sonra işi küçültüp yalnızca 10 dakikasını şimdi yap."
            )
            request = "10 dakika sonra ne yaptığını ya da seni gerçekten neyin durdurduğunu yaz."
        else:
            avoidance_count = skipped + rescheduled
            history = (
                f" Bu işi {avoidance_count} kez atladın ya da erteledin. Bu seni başarısız "
                "yapmaz ama aynı döngünün kendiliğinden değişmeyeceğini gösterir."
                if avoidance_count >= 2
                else ""
            )
            gap = (
                f"Şu işi başka bir zamana aldın: '{task_title}'. Yeni tarih tek başına "
                f"işi kolaylaştırmayacak.{history}"
            )
            intervention = (
                "Bu kez sadece günü değil, nasıl başlayacağını da netleştir: yeni saat "
                "geldiğinde yapacağın ilk küçük adımı şimdiden seç."
            )
            request = "Yeni zamanı ve o anda atacağın ilk adımı bana yaz."

        return NeuroFeedback(
            observed_evidence=(
                f"Güncel davranış olayı: {action.value}. "
                f"İlişkili kayıt sayısı: {evidence.total_events}."
            ),
            behavioral_pattern=pattern,
            interpretation=interpretation,
            neuro_context=(
                "Tekrarlanan ipucu–eylem eşleşmeleri öğrenme süreçlerini destekleyebilir; "
                "bu sistem kişisel dopamin veya korteks değişimi ölçmez."
            ),
            word_action_gap=gap,
            immediate_intervention=intervention,
            follow_up_minutes=12 if action == TaskAction.STARTED else 10,
            evidence_request=request,
            confidence=confidence,
            evidence_refs=evidence.evidence_refs[:8],
        )


class ResponsesIntelligenceProvider:
    """Shared structured-output implementation for Responses-compatible APIs."""

    def __init__(self, *, client: AsyncOpenAI, model: str) -> None:
        self.model = model
        self.client = client

    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan:
        response = await self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            safety_identifier=self._safety_identifier(user_id),
            input=[
                {
                    "role": "developer",
                    "content": DAILY_PLANNER_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "plan_date": plan_date,
                            "tasks": [task.model_dump(mode="json") for task in tasks],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            text_format=DailyPlan,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no structured daily plan.")
        plan = response.output_parsed
        return plan.model_copy(update={"generated_by": self.model})

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback:
        response = await self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "medium"},
            safety_identifier=self._safety_identifier(event.user_id),
            input=[
                {
                    "role": "developer",
                    "content": NEURO_FEEDBACK_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "event": event.model_dump(mode="json"),
                            "task_title": task_title,
                            "evidence": evidence.model_dump(mode="json"),
                            "critique_notes": critique_notes,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            text_format=NeuroFeedback,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no structured feedback.")
        return response.output_parsed

    @staticmethod
    def _safety_identifier(user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()[:32]


class OpenAIIntelligenceProvider(ResponsesIntelligenceProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        super().__init__(
            client=AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                max_retries=2,
                timeout=45,
            ),
            model=settings.openai_model,
        )


class GroqIntelligenceProvider(ResponsesIntelligenceProvider):
    """Use Groq's OpenAI-compatible Responses API on the cardless free tier."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required.")
        super().__init__(
            client=AsyncOpenAI(
                api_key=settings.groq_api_key.get_secret_value(),
                base_url="https://api.groq.com/openai/v1",
                max_retries=2,
                timeout=45,
            ),
            model=settings.groq_model,
        )


class ResilientIntelligenceProvider:
    """Keep the workflow useful when a free external LLM is unavailable or rate-limited."""

    def __init__(
        self,
        primary: IntelligenceProvider,
        fallback: IntelligenceProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan:
        try:
            return await self.primary.build_daily_plan(
                tasks=tasks,
                plan_date=plan_date,
                user_id=user_id,
            )
        except Exception as error:
            await logger.awarning(
                "llm_daily_plan_fallback",
                provider=type(self.primary).__name__,
                error_type=type(error).__name__,
            )
            return await self.fallback.build_daily_plan(
                tasks=tasks,
                plan_date=plan_date,
                user_id=user_id,
            )

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback:
        try:
            return await self.primary.generate_feedback(
                event=event,
                task_title=task_title,
                evidence=evidence,
                critique_notes=critique_notes,
            )
        except Exception as error:
            await logger.awarning(
                "llm_feedback_fallback",
                provider=type(self.primary).__name__,
                error_type=type(error).__name__,
            )
            return await self.fallback.generate_feedback(
                event=event,
                task_title=task_title,
                evidence=evidence,
                critique_notes=critique_notes,
            )


DAILY_PLANNER_PROMPT = """
You are the Planner Agent in a single-user accountability system.

Outcome:
- Turn today's Notion commitments into a realistic ordered DailyPlan.
- Preserve every task exactly once and preserve each task_id.
- Prioritize Core commitments, then P1 priority, then scheduled time.
- Use cognitive load and estimated duration to avoid an unrealistic sequence.
- Do not invent tasks, dates, evidence, diagnoses, or personal history.
- If capacity is excessive, keep the tasks but populate capacity_warning.
- Write headline in warm, conversational Turkish. Avoid bureaucratic terms such as
  taahhüt, kanıt, analiz, and optimizasyon in user-facing prose.

Return only the required structured DailyPlan.
""".strip()


NEURO_FEEDBACK_PROMPT = """
You are a direct neuro-behavioral strategist. Produce evidence-grounded feedback
about the user's current task event.

Required behavior:
- Start from supplied observed events and counts.
- Treat user self-report as self-report and model interpretation as a hypothesis.
- If has_sufficient_history is false, explicitly say that no reliable personal
  pattern can yet be concluded.
- State the word-action gap directly without generic praise or reassurance.
- Give one immediate action that can be executed now and one evidence request.
- Make word_action_gap, immediate_intervention, and evidence_request form one concise,
  vivid Turkish paragraph when joined in that order. Use plain language and no headings.
- Speak like a warm, honest guide accompanying the user through the day. Notice real
  progress, offer perspective after setbacks, and explain the next step in everyday Turkish.
- Avoid bureaucratic or clinical user-facing terms such as taahhüt, davranış kanıtı,
  söz-eylem açığı, müdahale, and güven skoru.
- Support autonomy and competence: connect confidence to the user's own observable action,
  not to praise, approval, obedience, or attachment to the assistant.
- Prefer a concrete cue-action or if-then plan over abstract motivation.
- Neuro context may explain learning, prediction, cognitive control, and habit
  mechanisms only in calibrated general language.

Forbidden:
- Claims that the system measured dopamine, receptor density, executive dysfunction,
  prefrontal cortex strength, neuroplastic change, or any clinical condition.
- Shame, humiliation, threats, dependency language, or claims that the user is weak.
- Covert manipulation, guaranteed transformation, or promises to raise dopamine.
- Fabricated counts, causes, memories, or certainty.

Incorporate critique_notes when present. Return only the required structured output.
""".strip()
