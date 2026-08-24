from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

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
                    f"{task.commitment_tier} taahhüt ve {task.priority} öncelik "
                    "sırasına göre konumlandırıldı."
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
            headline=f"{len(items)} taahhüt için kanıta dayalı günlük plan",
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
            gap = f"Verdiğin sözü tuttun: '{task_title}' tamamlandı."
            completion_label = "ilk" if completed <= 1 else f"{completed}."
            intervention = (
                "Kendine güvenmek için gereken kanıtı sen ürettin; bu, kayıtlardaki "
                f"{completion_label} somut tamamlaman."
            )
            request = (
                "Çıktıyı paylaş ve başlamanı sağlayan ipucunu tek cümleyle kaydet; "
                "aynı ipucunu yeniden kullan."
            )
        elif action == TaskAction.STARTED:
            gap = f"'{task_title}' için niyet bitti; davranış başladı."
            intervention = (
                "Disiplin, doğru duyguyu beklemek değil, başladığın küçük eylemi "
                "sürdürmektir: Minimum Action üzerinde şimdi 12 dakika kesintisiz çalış."
            )
            request = "Süre bitince sonucu 'Tamamladım' veya 'Engellendim' olarak bildir."
        elif action == TaskAction.BLOCKED:
            gap = f"'{task_title}' için söz–eylem zinciri şu anda engel noktasında duruyor."
            intervention = (
                "Engel son karar değil, çözülecek veridir: onu 'belirsizlik', "
                "'dış bağımlılık' veya 'başlangıç direnci' olarak sınıflandır ve "
                "yapılabilecek en küçük fiziksel adımı şimdi başlat."
            )
            request = "Engelin türünü ve attığın ilk adımı yaz."
        elif action == TaskAction.SKIPPED:
            avoidance_count = skipped + rescheduled
            history = (
                f" Bu görevdeki atlama/erteleme kaydı {avoidance_count}'e çıktı; "
                "bu sayı kimliğin değil, değiştireceğin döngünün kanıtı."
                if avoidance_count >= 2
                else ""
            )
            gap = f"'{task_title}' bugün atlandı; söz–eylem açığı açık kaldı.{history}"
            intervention = (
                "Bunu bir kimlik yargısına çevirmek yerine bir sonraki seçimi değiştir: "
                "yeni tarih vermeden önce 10 dakikalık Minimum Action'ı şimdi uygula."
            )
            request = "10 dakika sonra yapılan işi veya doğrulanabilir engeli bildir."
        else:
            avoidance_count = skipped + rescheduled
            history = (
                f" Bu görevdeki atlama/erteleme kaydı {avoidance_count}'e çıktı; "
                "bu sayı kimliğin değil, değiştireceğin döngünün kanıtı."
                if avoidance_count >= 2
                else ""
            )
            gap = (
                f"'{task_title}' yeniden planlandı; tarih değişti ama verilen söz henüz "
                f"davranışla kapanmadı.{history}"
            )
            intervention = (
                "Yeni tarihi kaçışa değil uygulanabilirliğe dönüştür: ilk 10 dakikalık "
                "Minimum Action'ı şimdi tamamla."
            )
            request = "Yaptığın somut işi ve yeni tarih için belirlediğin başlangıç ipucunu yaz."

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


class OpenAIIntelligenceProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.model = settings.openai_model
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=2,
            timeout=45,
        )

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


DAILY_PLANNER_PROMPT = """
You are the Planner Agent in a single-user accountability system.

Outcome:
- Turn today's Notion commitments into a realistic ordered DailyPlan.
- Preserve every task exactly once and preserve each task_id.
- Prioritize Core commitments, then P1 priority, then scheduled time.
- Use cognitive load and estimated duration to avoid an unrealistic sequence.
- Do not invent tasks, dates, evidence, diagnoses, or personal history.
- If capacity is excessive, keep the tasks but populate capacity_warning.

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
