from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extended')
def extended():
    return render_template('extended.html')


@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
