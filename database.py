import sqlite3
from datetime import datetime
import os

DB_PATH = 'database.db'

def init_db():
    """Initialize database with tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table untuk users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            birth_date TEXT,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height REAL,
            bmi REAL,
            medical_history TEXT,
            family_history TEXT,
            outdoor_activity TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Table untuk uploads (with user_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            original_filename TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT,
            reviewed INTEGER DEFAULT 0,
            actual_class TEXT,
            notes TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            reviewed_by TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Table untuk admin users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default admin (password: admin123)
    cursor.execute('''
        INSERT OR IGNORE INTO admin_users (username, password) 
        VALUES ('admin', 'admin123')
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully!")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================
# USER MANAGEMENT
# ============================================

def create_user(username, password, full_name, birth_date=None, gender=None, 
                weight=None, height=None, medical_history=None, 
                family_history=None, outdoor_activity=None):
    """Create new user account"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Calculate age from birth_date if provided
    age = None
    if birth_date:
        from datetime import datetime
        birth = datetime.strptime(birth_date, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    
    # Calculate BMI if weight and height provided
    bmi = None
    if weight and height:
        height_m = height / 100  # Convert cm to m
        bmi = weight / (height_m ** 2)
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password, full_name, birth_date, age, 
                             gender, weight, height, bmi, medical_history, 
                             family_history, outdoor_activity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, password, full_name, birth_date, age, gender, weight, 
              height, bmi, medical_history, family_history, outdoor_activity))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None  # Username already exists

def verify_user(username, password):
    """Verify user credentials and return user data"""
    conn = get_db_connection()
    
    user = conn.execute(
        'SELECT * FROM users WHERE username = ? AND password = ?',
        (username, password)
    ).fetchone()
    
    if user:
        # Update last login
        conn.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now(), user['id'])
        )
        conn.commit()
    
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    """Get user data by ID"""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def update_user_profile(user_id, **kwargs):
    """Update user profile data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build UPDATE query dynamically
    fields = []
    values = []
    
    for key, value in kwargs.items():
        if value is not None:
            fields.append(f"{key} = ?")
            values.append(value)
    
    # Recalculate BMI if weight or height changed
    if 'weight' in kwargs or 'height' in kwargs:
        user = get_user_by_id(user_id)
        weight = kwargs.get('weight', user.get('weight'))
        height = kwargs.get('height', user.get('height'))
        
        if weight and height:
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            fields.append("bmi = ?")
            values.append(bmi)
    
    if fields:
        values.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
        cursor.execute(query, values)
        conn.commit()
    
    conn.close()

# ============================================
# UPLOAD MANAGEMENT
# ============================================

def save_upload(filename, original_filename, predicted_class, confidence, risk_level, user_id=None):
    """Save upload record to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO uploads (filename, original_filename, predicted_class, 
                           confidence, risk_level, user_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, original_filename, predicted_class, confidence, risk_level, user_id))
    
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return upload_id

def get_all_uploads(reviewed=None):
    """Get all uploads, optionally filter by reviewed status"""
    conn = get_db_connection()
    
    if reviewed is None:
        uploads = conn.execute('''
            SELECT u.*, users.full_name, users.age, users.gender, users.bmi
            FROM uploads u
            LEFT JOIN users ON u.user_id = users.id
            ORDER BY u.uploaded_at DESC
        ''').fetchall()
    else:
        uploads = conn.execute('''
            SELECT u.*, users.full_name, users.age, users.gender, users.bmi
            FROM uploads u
            LEFT JOIN users ON u.user_id = users.id
            WHERE u.reviewed = ? 
            ORDER BY u.uploaded_at DESC
        ''', (1 if reviewed else 0,)).fetchall()
    
    conn.close()
    return uploads

def get_user_uploads(user_id):
    """Get all uploads for a specific user"""
    conn = get_db_connection()
    uploads = conn.execute(
        'SELECT * FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return uploads

def get_upload_by_id(upload_id):
    """Get single upload by ID"""
    conn = get_db_connection()
    upload = conn.execute('SELECT * FROM uploads WHERE id = ?', (upload_id,)).fetchone()
    conn.close()
    return upload

def update_upload_review(upload_id, actual_class, notes, reviewed_by):
    """Update upload with review information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE uploads 
        SET reviewed = 1, 
            actual_class = ?, 
            notes = ?,
            reviewed_at = ?,
            reviewed_by = ?
        WHERE id = ?
    ''', (actual_class, notes, datetime.now(), reviewed_by, upload_id))
    
    conn.commit()
    conn.close()

def delete_upload(upload_id):
    """Delete upload record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM uploads WHERE id = ?', (upload_id,))
    conn.commit()
    conn.close()

def get_statistics():
    """Get statistics for dashboard"""
    conn = get_db_connection()
    
    stats = {
        'total_uploads': conn.execute('SELECT COUNT(*) FROM uploads').fetchone()[0],
        'total_users': conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
        'pending_review': conn.execute('SELECT COUNT(*) FROM uploads WHERE reviewed = 0').fetchone()[0],
        'reviewed': conn.execute('SELECT COUNT(*) FROM uploads WHERE reviewed = 1').fetchone()[0],
    }
    
    # Get prediction accuracy (where reviewed)
    reviewed_total = conn.execute('SELECT COUNT(*) FROM uploads WHERE reviewed = 1').fetchone()[0]
    correct_predictions = conn.execute('''
        SELECT COUNT(*) FROM uploads 
        WHERE reviewed = 1 AND predicted_class = actual_class
    ''').fetchone()[0]
    
    stats['accuracy'] = (correct_predictions / reviewed_total * 100) if reviewed_total > 0 else 0
    
    # Demographic stats
    stats['avg_age'] = conn.execute('''
        SELECT AVG(users.age) FROM uploads 
        JOIN users ON uploads.user_id = users.id
    ''').fetchone()[0] or 0
    
    stats['avg_bmi'] = conn.execute('''
        SELECT AVG(users.bmi) FROM uploads 
        JOIN users ON uploads.user_id = users.id
    ''').fetchone()[0] or 0
    
    conn.close()
    return stats

def get_demographic_analysis():
    """Get demographic analysis for research"""
    conn = get_db_connection()
    
    analysis = {}
    
    # Age distribution
    analysis['age_distribution'] = conn.execute('''
        SELECT 
            CASE 
                WHEN age < 20 THEN '< 20'
                WHEN age BETWEEN 20 AND 30 THEN '20-30'
                WHEN age BETWEEN 31 AND 40 THEN '31-40'
                WHEN age BETWEEN 41 AND 50 THEN '41-50'
                WHEN age BETWEEN 51 AND 60 THEN '51-60'
                ELSE '> 60'
            END as age_group,
            COUNT(*) as count,
            SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count
        FROM uploads
        JOIN users ON uploads.user_id = users.id
        WHERE users.age IS NOT NULL
        GROUP BY age_group
        ORDER BY age_group
    ''').fetchall()
    
    # Gender distribution
    analysis['gender_distribution'] = conn.execute('''
        SELECT 
            gender,
            COUNT(*) as count,
            SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count
        FROM uploads
        JOIN users ON uploads.user_id = users.id
        WHERE gender IS NOT NULL
        GROUP BY gender
    ''').fetchall()
    
    # BMI correlation
    analysis['bmi_correlation'] = conn.execute('''
        SELECT 
            CASE 
                WHEN bmi < 18.5 THEN 'Underweight'
                WHEN bmi BETWEEN 18.5 AND 24.9 THEN 'Normal'
                WHEN bmi BETWEEN 25 AND 29.9 THEN 'Overweight'
                ELSE 'Obese'
            END as bmi_category,
            COUNT(*) as count,
            SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_count
        FROM uploads
        JOIN users ON uploads.user_id = users.id
        WHERE bmi IS NOT NULL
        GROUP BY bmi_category
    ''').fetchall()
    
    conn.close()
    return analysis

def verify_admin(username, password):
    """Verify admin credentials"""
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM admin_users WHERE username = ? AND password = ?',
        (username, password)
    ).fetchone()
    conn.close()
    return user is not None

# Initialize database on import
if __name__ == '__main__':
    init_db()