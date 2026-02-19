import asyncio
import os
import discord
import uvloop

# Install uvloop before anything else
uvloop.install()

# --- Configuration ---
# Load credentials from environment variables
BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GUILD_ID = int(os.environ.get("DISCORD_GUILD_ID", 0))
VOICE_CHANNEL_ID = int(os.environ.get("DISCORD_VOICE_CHANNEL_ID", 0))

# --- Bot Code ---

async def run_test():
    """Initializes and runs the bot within an async context."""
    intents = discord.Intents.default()
    intents.guilds = True
    intents.voice_states = True
    bot = discord.Bot(intents=intents)

    @bot.event
    async def on_ready():
        print(f"Бот {bot.user} запущен и готов.")
        print("---")
        print(f"Целевой сервер (Guild ID): {GUILD_ID}")
        print(f"Целевой канал (Voice Channel ID): {VOICE_CHANNEL_ID}")
        print("---")

        guild = bot.get_guild(GUILD_ID)
        if not guild:
            print(f"Ошибка: Не удалось найти сервер с ID: {GUILD_ID}")
            await bot.close()
            return

        channel = guild.get_channel(VOICE_CHANNEL_ID)
        if not channel or not isinstance(channel, discord.VoiceChannel):
            print(f"Ошибка: Не удалось найти голосовой канал с ID: {VOICE_CHANNEL_ID}")
            await bot.close()
            return

        print(f"Попытка подключения к голосовому каналу: '{channel.name}'...")
        voice_client = None
        try:
            # Connect to the voice channel with a long timeout
            voice_client = await channel.connect(timeout=60.0)

            # is_connected() can sometimes be slow to update, let's wait a moment
            await asyncio.sleep(1)

            if voice_client.is_connected():
                print("\n✅ УСПЕХ! Голосовое соединение установлено!")
                print(f"Статус клиента: {voice_client.is_connected()}")
                print(f"Пинг: {voice_client.latency * 1000:.2f}ms")
            else:
                print("\n❌ ПРОВАЛ! Соединение было установлено, но is_connected() равно False.")

        except asyncio.TimeoutError:
            print("\n❌ ПРОВАЛ! Ошибка: asyncio.TimeoutError. Соединение заняло слишком много времени.")
        except Exception as e:
            print(f"\n❌ ПРОВАЛ! Произошла непредвиденная ошибка: {e}")
        finally:
            if voice_client and voice_client.is_connected():
                print("\nОтключаюсь...")
                await voice_client.disconnect()
                print("Отключился.")
            
            print("Завершение работы бота.")
            await bot.close()

    # We use bot.start() here because bot.run() blocks and manages the loop,
    # but we are already managing the loop with asyncio.run().
    await bot.start(BOT_TOKEN)

def main():
    if not all([BOT_TOKEN, GUILD_ID, VOICE_CHANNEL_ID]):
        print("Ошибка: Не установлены все необходимые переменные окружения.")
        print("Пожалуйста, установите DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, и DISCORD_VOICE_CHANNEL_ID.")
        return

    print("Запуск тестового бота с uvloop...")
    try:
        # asyncio.run() creates and manages the event loop for us.
        asyncio.run(run_test())
    except discord.errors.LoginFailure:
        print("ПРОВАЛ! Неверный токен бота. Пожалуйста, проверьте DISCORD_BOT_TOKEN.")
    except Exception as e:
        print(f"ПРОВАЛ! Ошибка на верхнем уровне: {e}")

if __name__ == "__main__":
    main()