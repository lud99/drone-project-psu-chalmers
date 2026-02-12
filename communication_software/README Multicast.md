# Setup

To get backend multicasting to work, you need to:
1. ```cd communication_software```
2. ```pip install python-dotenv```
3. The first time, or every time your ip changes, (for example if you switch networks or it changes periodically), run
```python export_host_ip.py```
4. Start the forwarded program in a seperate terminal window with ```python forward_multicast.py```
