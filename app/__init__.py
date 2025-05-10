from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Import and register blueprints
    from app.routes import bp
    app.register_blueprint(bp)
    
    # Configure app
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['JSON_SORT_KEYS'] = False
    
    return app

