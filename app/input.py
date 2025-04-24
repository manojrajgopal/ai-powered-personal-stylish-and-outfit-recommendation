import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.virtual_try_on import run_virtual_try_on
username = session['user']['username']
vton_img_path = user_info[2] 
garm_img_path = ""

# Run the virtual try-on process
one, two = run_virtual_try_on(username, vton_img_path, garm_img_path)

# Print the output result
print(one, two)