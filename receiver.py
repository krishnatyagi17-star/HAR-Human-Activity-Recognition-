from flask import Flask, request
import json
import time

app = Flask(__name__)
count = 0


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def receive(path):
    global count
    if request.method == 'GET':
        return 'receiver is running'

    count += 1
    body = request.get_data(as_text=True)

    with open('stream_log.jsonl', 'a') as f:
        f.write(body.replace('\n', ' ') + '\n')

    print('---- message', count, 'at', time.strftime('%H:%M:%S'), 'size', len(body), 'characters')
    try:
        data = json.loads(body)
        print('top-level keys:', list(data.keys()))
        items = data.get('payload', [])
        counts = {}
        for item in items:
            name = item.get('name', '?')
            counts[name] = counts.get(name, 0) + 1
        print('readings per sensor in this message:', counts)
        if count == 1:
            print('first message, first 600 characters:')
            print(body[:600])
    except Exception as e:
        print('could not read as JSON:', e)
        print(body[:300])

    return 'ok'


if __name__ == '__main__':
    print('Waiting for the phone... (press Ctrl + C to stop)')
    app.run(host='0.0.0.0', port=8000)
