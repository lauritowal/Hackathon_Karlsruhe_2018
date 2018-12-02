from app import app


@app.route('/')
@app.route('/test/', methods=['POST', 'GET'])
def test():
    return "Hello, World!"