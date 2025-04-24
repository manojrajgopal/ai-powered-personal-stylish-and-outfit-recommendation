import pymysql

class DatabaseInitializer:
    def __init__(self, host='localhost', user='username', password='password', database='fashion'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the MySQL server."""
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password
        )
        self.cursor = self.connection.cursor()

    def create_database(self):
        """Create the 'fashion' database if it doesn't exist."""
        self.cursor.execute("SHOW DATABASES LIKE %s", (self.database,))
        database_exists = self.cursor.fetchone()

        if not database_exists:
            self.cursor.execute(f"CREATE DATABASE {self.database}")

    def use_database(self):
        """Select the 'fashion' database for use."""
        self.cursor.execute(f"USE {self.database}")

    def create_login_table(self):

        """Create the 'login' table if it doesn't exist."""
        self.cursor.execute("SHOW TABLES LIKE 'login'")
        login_table_exists = self.cursor.fetchone()

        if not login_table_exists:
            self.cursor.execute("""
                CREATE TABLE login (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    phone VARCHAR(20),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            """)

    def create_user_information_table(self):
        """Create the 'user_information' table if it doesn't exist with ON DELETE CASCADE."""
        self.cursor.execute("SHOW TABLES LIKE 'user_information'")
        user_info_table_exists = self.cursor.fetchone()

        if not user_info_table_exists:
            self.cursor.execute("""
                CREATE TABLE user_information (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    profile_pic VARCHAR(255),
                    gender VARCHAR(10),
                    date_of_birth DATE,
                    body_type VARCHAR(50),
                    height FLOAT,
                    weight FLOAT,
                    preferred_color VARCHAR(50),
                    preferred_fabrics VARCHAR(255),
                    preferred_styles VARCHAR(255),
                    occasion_types VARCHAR(255),
                    style_goals VARCHAR(255),
                    budget FLOAT,
                    skin_color VARCHAR(50),
                    wardrobe_img VARCHAR(255),
                    user_title VARCHAR(50),
                    user_about_1 TEXT,
                    user_about_2 TEXT,
                    virtual_try_on_image VARCHAR(255),
                    CONSTRAINT fk_user_info_username 
                        FOREIGN KEY (username) REFERENCES login(username) 
                        ON DELETE CASCADE
                )
            """)


    def close_connection(self):
        """Close the cursor and connection."""
        self.cursor.close()
        self.connection.close()

    def initialize_database(self):
        """Run all database setup tasks."""
        self.connect()
        self.create_database()
        self.use_database()
        self.create_login_table()
        self.create_user_information_table()
        self.close_connection()


# Usage of the class in main.py
if __name__ == '__main__':
    db_initializer = DatabaseInitializer()
    db_initializer.initialize_database()