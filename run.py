import os
from app import create_app
from flask import Flask

app = create_app(None)

@app.shell_context_processor
def make_shell_context():
    return dict()

if __name__ == '__main__':
    Flask.run(app, debug=True, host='0.0.0.0')