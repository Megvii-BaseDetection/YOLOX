import socketio

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')


@sio.event
def detection_result(data):
    print('Received message ', data['timestamp'])


@sio.event
def disconnect():
    print('disconnected from server')


sio.connect('http://localhost:1234')
sio.emit('run_detection', {'rtmp_url': "rtmp://localhost/show/stream"})
