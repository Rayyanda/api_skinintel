from flask import Flask, request, jsonify, session, render_template, redirect, url_for, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import database
import zipfile
from functools import wraps

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-change-this-in-production'  # Ganti dengan random string

# Configuration
UPLOAD_FOLDER = 'uploads'
PENDING_FOLDER = os.path.join(UPLOAD_FOLDER, 'pending')
REVIEWED_FOLDER = os.path.join(UPLOAD_FOLDER, 'reviewed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders if not exist
os.makedirs(PENDING_FOLDER, exist_ok=True)
os.makedirs(REVIEWED_FOLDER, exist_ok=True)

# Initialize database
database.init_db()

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'model', 'labels.txt')

# Load TFLite model
print("=" * 50)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_image(image, original_filename, predicted_class, confidence, risk_level, user_id=None):
    """Save uploaded image to pending folder and database"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
        filename = f"{timestamp}_{secure_filename(original_filename)}"
        
        # Save image
        filepath = os.path.join(PENDING_FOLDER, filename)
        image.save(filepath)
        
        # Save to database with user_id
        upload_id = database.save_upload(
            filename=filename,
            original_filename=original_filename,
            predicted_class=predicted_class,
            confidence=confidence,
            risk_level=risk_level,
            user_id=user_id
        )
        
        return upload_id, filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None, None

print("=" * 50)
print("Loading TFLite model...")
print(f"Model path: {MODEL_PATH}")
print(f"Labels path: {LABELS_PATH}")

try:
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✓ Model loaded successfully!")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    model_loaded = True
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    interpreter = None
    input_details = None
    output_details = None
    model_loaded = False

# Load labels
try:
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Labels loaded: {class_names}")
except Exception as e:
    print(f"✗ Error loading labels: {e}")
    class_names = []

print("=" * 50)

def preprocess_image(image):
    """Preprocess image untuk TFLite model"""
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [-1, 1] (Teachable Machine format)
    normalized_image = (img_array / 127.5) - 1
    
    # Add batch dimension
    input_data = np.expand_dims(normalized_image, axis=0)
    
    return input_data

def predict_image(image):
    """Predict image dengan TFLite model"""
    if not model_loaded or interpreter is None:
        raise Exception("Model not loaded")
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    
    # Get predictions
    predictions = []
    for i, score in enumerate(results):
        predictions.append({
            'class': class_names[i] if i < len(class_names) else f'Class {i}',
            'confidence': float(score)
        })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

@app.route('/')
def home():
    """Homepage endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Skin Cancer Detection API',
        'version': '2.0',
        'model_type': 'TensorFlow Lite',
        'model_loaded': model_loaded,
        'classes': class_names,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint
    Accepts: 
    - multipart/form-data dengan 'image' file
    - application/json dengan 'image_base64' field
    """
    try:
        # Check model
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        original_filename = 'unknown.jpg'
        image_to_save = None
        
        # Get image from request
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            
            # Validate file
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            original_filename = file.filename
            
            # Read image
            try:
                image = Image.open(file.stream)
                # Save copy untuk database
                image_to_save = image.copy()
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid image file: {str(e)}'
                }), 400
            
        elif request.is_json and 'image_base64' in request.json:
            # Base64 upload
            try:
                image_data = base64.b64decode(request.json['image_base64'])
                image = Image.open(io.BytesIO(image_data))
                image_to_save = image.copy()
                original_filename = request.json.get('filename', 'image.jpg')
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid base64 image: {str(e)}'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send "image" file or "image_base64" in JSON'
            }), 400
        
        # Predict
        predictions = predict_image(image)
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No predictions returned'
            }), 500
        
        top_prediction = predictions[0]
        
        # Determine risk level
        confidence = top_prediction['confidence']
        class_name = top_prediction['class'].lower()
        
        # Risk assessment logic
        if 'malignant' in class_name or 'melanoma' in class_name or 'cancer' in class_name:
            if confidence > 0.7:
                risk_level = 'high'
            elif confidence > 0.5:
                risk_level = 'medium'
            else:
                risk_level = 'low'
        elif 'benign' in class_name or 'normal' in class_name:
            if confidence > 0.8:
                risk_level = 'low'
            else:
                risk_level = 'medium'
        else:
            risk_level = 'medium'
        
        # Save uploaded image to database
        upload_id = None
        if image_to_save:
            # Get user_id from request (if logged in)
            user_id = None
            if request.is_json and 'user_id' in request.json:
                user_id = request.json['user_id']
            elif 'user_id' in request.form:
                user_id = request.form.get('user_id')
            
            upload_id, saved_filename = save_uploaded_image(
                image_to_save,
                original_filename,
                top_prediction['class'],
                confidence,
                risk_level,
                user_id  # Pass user_id
            )
        
        # Response
        response_data = {
            'success': True,
            'prediction': {
                'class': top_prediction['class'],
                'confidence': top_prediction['confidence'],
                'confidence_percentage': f"{top_prediction['confidence'] * 100:.1f}%",
                'risk_level': risk_level
            },
            'all_predictions': predictions,
            'disclaimer': 'Hasil ini BUKAN diagnosis medis. Segera konsultasikan dengan dokter spesialis kulit untuk pemeriksaan lebih lanjut.'
        }
        
        if upload_id:
            response_data['upload_id'] = upload_id
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_type': 'TensorFlow Lite',
        'classes_count': len(class_names)
    })

# ============================================
# USER AUTH ROUTES
# ============================================

@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['username', 'password', 'full_name']
        for field in required:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create user
        user_id = database.create_user(
            username=data['username'],
            password=data['password'],  # NOTE: Hash in production!
            full_name=data['full_name'],
            birth_date=data.get('birth_date'),
            gender=data.get('gender'),
            weight=data.get('weight'),
            height=data.get('height'),
            medical_history=data.get('medical_history'),
            family_history=data.get('family_history'),
            outdoor_activity=data.get('outdoor_activity')
        )
        
        if user_id is None:
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 409
        
        # Get user data
        user = database.get_user_by_id(user_id)
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'full_name': user['full_name'],
                'age': user['age'],
                'gender': user['gender'],
                'bmi': user['bmi']
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password required'
            }), 400
        
        user = database.verify_user(username, password)
        
        if user is None:
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'full_name': user['full_name'],
                'age': user['age'],
                'gender': user['gender'],
                'weight': user['weight'],
                'height': user['height'],
                'bmi': user['bmi'],
                'medical_history': user['medical_history'],
                'family_history': user['family_history'],
                'outdoor_activity': user['outdoor_activity']
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user profile"""
    try:
        user = database.get_user_by_id(user_id)
        
        if user is None:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'full_name': user['full_name'],
                'age': user['age'],
                'gender': user['gender'],
                'weight': user['weight'],
                'height': user['height'],
                'bmi': user['bmi'],
                'medical_history': user['medical_history'],
                'family_history': user['family_history'],
                'outdoor_activity': user['outdoor_activity'],
                'created_at': user['created_at'],
                'last_login': user['last_login']
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<int:user_id>/history', methods=['GET'])
def get_user_history(user_id):
    """Get user's prediction history"""
    try:
        uploads = database.get_user_uploads(user_id)
        
        history = []
        for upload in uploads:
            history.append({
                'id': upload['id'],
                'predicted_class': upload['predicted_class'],
                'confidence': upload['confidence'],
                'risk_level': upload['risk_level'],
                'uploaded_at': upload['uploaded_at'],
                'reviewed': bool(upload['reviewed']),
                'actual_class': upload['actual_class']
            })
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ADMIN ROUTES
# ============================================

def login_required(f):
    """Decorator to require admin login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if database.verify_admin(username, password):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard"""
    stats = database.get_statistics()
    recent_uploads = database.get_all_uploads()[:10]  # Latest 10
    return render_template('admin_dashboard.html', stats=stats, recent_uploads=recent_uploads)

@app.route('/admin/review')
@login_required
def admin_review():
    """Review pending uploads"""
    pending = database.get_all_uploads(reviewed=False)
    return render_template('admin_review.html', uploads=pending, class_names=class_names)

@app.route('/admin/review/<int:upload_id>', methods=['POST'])
@login_required
def admin_review_submit(upload_id):
    """Submit review for an upload"""
    actual_class = request.form.get('actual_class')
    notes = request.form.get('notes', '')
    action = request.form.get('action')
    
    if action == 'approve':
        database.update_upload_review(
            upload_id,
            actual_class,
            notes,
            session['admin_username']
        )
        
        # Move file to reviewed folder
        upload = database.get_upload_by_id(upload_id)
        if upload:
            old_path = os.path.join(PENDING_FOLDER, upload['filename'])
            
            # Create class folder if not exists
            class_folder = os.path.join(REVIEWED_FOLDER, actual_class)
            os.makedirs(class_folder, exist_ok=True)
            
            new_path = os.path.join(class_folder, upload['filename'])
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
    
    elif action == 'delete':
        upload = database.get_upload_by_id(upload_id)
        if upload:
            # Delete file
            filepath = os.path.join(PENDING_FOLDER, upload['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Delete from database
            database.delete_upload(upload_id)
    
    return redirect(url_for('admin_review'))

@app.route('/admin/export')
@login_required
def admin_export():
    """Export reviewed dataset as ZIP"""
    try:
        # Create ZIP file
        zip_filename = f'dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add all reviewed images
            for class_name in os.listdir(REVIEWED_FOLDER):
                class_path = os.path.join(REVIEWED_FOLDER, class_name)
                if os.path.isdir(class_path):
                    for filename in os.listdir(class_path):
                        file_path = os.path.join(class_path, filename)
                        arcname = os.path.join(class_name, filename)
                        zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True)
    
    except Exception as e:
        return f"Error exporting dataset: {str(e)}", 500

@app.route('/admin/image/<folder>/<filename>')
@login_required
def admin_image(folder, filename):
    """Serve image for admin review"""
    if folder == 'pending':
        filepath = os.path.join(PENDING_FOLDER, filename)
    else:
        filepath = os.path.join(REVIEWED_FOLDER, folder, filename)
    
    return send_file(filepath)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Development server
    print("\nStarting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print("Press CTRL+C to quit\n")
    
    # Get port from environment (for Render.com)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)