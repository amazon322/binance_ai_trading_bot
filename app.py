from flask import Flask
from config import Config
# Import routes
from app_routes.main_routes import main_routes  # main_routes is the Blueprint instance


def create_app():
    app = Flask(__name__)

    # Load configurations
    app.config.from_object(Config)

    # Secret key for session management
    app.secret_key = 'your_secret_key_here'  # Replace with a secure key

    # Register Blueprints (if any)
    # from yourmodule import your_blueprint
    # app.register_blueprint(your_blueprint)

    app.register_blueprint(main_routes)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
