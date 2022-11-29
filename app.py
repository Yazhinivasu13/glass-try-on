from flask import Flask, render_template, Response

from live_try_on import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames("./images/glasses.png"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/submit/<string:img_url>', methods=['POST', 'GET'])
def submit(img_url):
    img_url = "images/"+img_url
    return Response(generate_frames(img_url), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
