import asyncio
import os
import sys
from typing import Optional

from src.llm.prompts import DOS_PROMPT


def import_dos_modules():
    try:
        from src.consts import GAME_URL_MAP
        from src.emulators.dos.interface import DOSGameInterface
        from src.emulators.dos.website_server import DOSGameServer
        from src.llm.vgagent import WebBrowsingVGAgent
        from src.vgbench_evaluator import DOSEvaluator

        return (
            WebBrowsingVGAgent,
            GAME_URL_MAP,
            DOSGameServer,
            DOSEvaluator,
            DOSGameInterface,
        )
    except ImportError as e:
        print(f"Error importing DOS modules: {e}")
        print(
            "Make sure you have installed the required dependencies for DOS emulation."
        )
        sys.exit(1)


async def run_dos_emulator(args):
    """Run the DOS emulator with the given arguments."""
    # Import required modules
    WebBrowsingVGAgent, GAME_URL_MAP, DOSGameServer, DOSEvaluator, DOSGameInterface = (
        import_dos_modules()
    )

    # Set up variables
    task = args.task
    url = args.url
    model = args.model

    if model is None:
        print("No model provided, using default model: gpt-4o")
        model = "gpt-4o"

    api_key = args.api_key

    headless = args.headless
    temperature = args.temperature
    max_tokens = args.max_tokens
    dos_name = args.game
    website_only = args.website_only

    # Start a local server for DOS games if specified
    server = None
    if dos_name:
        server = DOSGameServer(args.port, lite=args.lite)

        # Use custom HTML if provided, otherwise fall back to GAME_URL_MAP
        if hasattr(args, "custom_html") and args.custom_html:
            url = server.start(dos_name, args.custom_html)
        elif dos_name in GAME_URL_MAP:
            dos_game = GAME_URL_MAP[dos_name]
            url = server.start(dos_game)
        else:
            print(f"Error: No configuration found for game {dos_name}")
            return

    if task == "":
        print("No task provided, using default task")
        task = DOS_PROMPT

    # Website-only mode - open the browser and keep server running
    if website_only:
        if url:
            print(f"Opening {url} in Chromium browser...")

            # Use Playwright to open in Chromium instead of system default browser
            await server.open_in_chromium(headless=headless)

            # Keep the server running until interrupted
            print("Press Ctrl+C to stop the server and exit.")
            try:
                # Wait indefinitely until interrupted
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("\nShutting down server...")
            finally:
                if server:
                    server.stop()
            return
        else:
            print("Error: No URL specified for website-only mode.")
            if server:
                server.stop()
            return

    if args.lite:
        print("Playing VideoGameBench Lite on DOS Emulator")
    else:
        print("Playing VideoGameBench on DOS Emulator")

    # Create the agent for interactive mode
    agent = WebBrowsingVGAgent(
        model=model,
        api_key=api_key,
        game=dos_name,
        headless=headless,
        temperature=temperature,
        max_tokens=max_tokens,
        context_window=args.max_context_size,
        lite=args.lite,
        enable_ui=args.enable_ui,
        task_prompt=args.task_prompt,
        press_key_delay=args.press_key_delay,
        api_base=args.api_base,
    )

    game_interface = DOSGameInterface(
        headless=headless,
        game=dos_name,
        lite=args.lite,
        num_screenshots_per_action=args.num_screenshots_per_action,
    )

    evaluator = DOSEvaluator(
        max_steps=args.max_steps,
        step_delay=args.step_delay,
        checkpoints=args.checkpoints,
        game_interface=game_interface,
        threshold=args.threshold,
    )

    await evaluator.start(url)
    await evaluator.run_episode(agent, task, server)
