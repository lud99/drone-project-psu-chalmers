#!/usr/bin/env python
import asyncio
from .main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[EXIT] Interrupted by user")
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
