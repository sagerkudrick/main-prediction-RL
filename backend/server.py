from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='../static', static_url_path='/static')

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('..', path)

if __name__ == '__main__':
    print("Starting static file server at http://localhost:8000")
    app.run(port=8000, debug=True)


