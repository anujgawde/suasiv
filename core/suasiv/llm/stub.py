from __future__ import annotations

from suasiv.llm.base import LLMBackend


class StubBackend(LLMBackend):
    name = "stub"

    def complete(self, system: str, prompt: str) -> str:
        return """## Overall Assessment

This presentation showed solid fundamentals with room for improvement in mid-section delivery.

## What Worked

- **Strong opening**: The speaker opened with high vocal energy and consistent eye contact, with 90% of audience members visually engaged during the first 7 seconds.
- **Clear topic signaling**: The transition to quarterly results was well-structured, with an audience member nodding in acknowledgment.

## What to Improve

- **Mid-section energy drop**: Between 11.0s and 14.0s, vocal energy decreased noticeably. During this same window, audience attention dropped to 40% and the speaker broke eye contact.
- **Filler words**: One filler word ("um") detected at 11.5s, coinciding with the energy drop. This signals a moment of uncertainty the audience may pick up on.

## Key Moments

1. **[0:00 - 3:30] Strong moment**: High energy opening with full audience engagement
2. **[11:00 - 14:00] Weak moment**: Energy drop + attention drop + filler word + gaze break
3. **[14:30 - 16:00] Q&A moment**: Audience member asked a question

## Audience Reception

- Overall attention: 72%
- Positive reactions: 60% of tracked moments
- One notable attention drop during the mid-section data presentation

## Next Steps

1. Practice the data-heavy section to maintain vocal energy through numbers
2. Prepare transition phrases to bridge from data to narrative sections
3. Maintain camera eye contact during less rehearsed segments
"""
