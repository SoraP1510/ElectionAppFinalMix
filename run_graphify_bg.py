import subprocess, os
os.chdir(r'D:\Download\AI\ElectionAppFinalMix')
subprocess.run(['taskkill', '/F', '/IM', 'graphify.exe'], capture_output=True)
log = open('graphify-log.txt', 'w', encoding='utf-8')
p = subprocess.Popen(
    ['graphify', 'extract', '.', '--backend', 'ollama', '--model', 'qwen3:latest', '--max-concurrency', '1', '--token-budget', '3000', '--api-timeout', '300'],
    stdout=log, stderr=subprocess.STDOUT, cwd=r'D:\Download\AI\ElectionAppFinalMix'
)
print(f"PID: {p.pid}")
