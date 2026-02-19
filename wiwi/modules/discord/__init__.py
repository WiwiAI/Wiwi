"""Discord модуль для Wiwi."""

from wiwi.modules.discord.module import DiscordModule
from wiwi.modules.discord.wake_word import WakeWordPipeline, WakeWordState
from wiwi.modules.discord.sounds import SoundEffects

__all__ = ["DiscordModule", "WakeWordPipeline", "WakeWordState", "SoundEffects"]
