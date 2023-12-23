from flask import Blueprint
from flask import Flask, render_template, request, jsonify, redirect

#views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return "home page"

@views.route('/profile')
def profile():
    args = request.args
    name = args.get('name')
    return render_template('index.html', name=name)

@views.route('/data')
def get_data():
    data = request.json
    return jsonify(data)
