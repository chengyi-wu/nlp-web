from flask import render_template
from . import main
import json

@main.route('/', methods=['GET', 'POST'])
def index():
    return json.dumps({
        'response': 200
    })