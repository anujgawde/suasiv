from suasiv.analyzers.transcript import TranscriptAnalyzer
from suasiv.analyzers.diarization import DiarizationAnalyzer
from suasiv.analyzers.pacing import PacingAnalyzer
from suasiv.analyzers.prosody import ProsodyAnalyzer
from suasiv.analyzers.content import ContentAnalyzer
from suasiv.analyzers.speaker_facial import SpeakerFacialAnalyzer
from suasiv.analyzers.audience_engagement import AudienceEngagementAnalyzer
from suasiv.analyzers.audience_reaction import AudienceReactionAnalyzer
from suasiv.analyzers.audience_verbal import AudienceVerbalAnalyzer

ALL_ANALYZERS = [
    TranscriptAnalyzer,
    DiarizationAnalyzer,
    PacingAnalyzer,
    ProsodyAnalyzer,
    ContentAnalyzer,
    SpeakerFacialAnalyzer,
    AudienceEngagementAnalyzer,
    AudienceReactionAnalyzer,
    AudienceVerbalAnalyzer,
]
