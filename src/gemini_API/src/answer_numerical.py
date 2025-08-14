#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import google.generativeai as genai
import os
import json
from PIL import Image

# --- Configuration ---
# IMPORTANT: Set your Google API Key as an environment variable
# export GOOGLE_API_KEY="YOUR_API_KEY"
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    rospy.logerr("GEMINI_API_KEY environment variable not set. Please export your API key.")
    exit()

genai.configure(api_key=API_KEY)

# --- File Paths ---
# Assumes this script is in ai_module/src/gemini_API
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODULE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..",".."))
RELTR_SG_DIR = os.path.abspath(os.path.join(AI_MODULE_DIR, "src","reltr_scene_graph","reltr_scene_graph"))

IMAGE_DIR = os.path.join(AI_MODULE_DIR, "data", "image_per_node")
SCENE_GRAPH_PATH = os.path.join(RELTR_SG_DIR, "data", "all_merged_sg", "all_merged_sg.json")

class NumericalAnswerGenerator:
    def __init__(self):
        rospy.init_node('answer_numerical_node', anonymous=True)

        # --- ROS Parameters for Mode Selection ---
        # 1: Image-only, Integer
        # 2: Image-only, Integer + Explanation
        # 3: Image + Scene Graph, Integer
        # 4: Image + Scene Graph, Integer + Explanation
        self.mode = rospy.get_param("~mode", 2)
        
        # --- ROS Communication ---
        # Subscriber for the question
        rospy.Subscriber("/challenge_question", String, self.question_callback)
        # Publisher for the answer
        self.response_pub = rospy.Publisher('/numerical_response', String, queue_size=10)

        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.images = self.load_images()
        self.scene_graph = self.load_scene_graph()

        rospy.loginfo(f"Numerical Answer Node started in Mode: {self.mode}")
        rospy.loginfo("Waiting for a question on /challenge_question...")

    def load_images(self):
        """Loads all PNG images from the specified directory."""
        images = []
        if not os.path.exists(IMAGE_DIR):
            rospy.logerr(f"Image directory not found at {IMAGE_DIR}")
            return images
        
        # Sort files numerically (e.g., 0.png, 1.png, 10.png)
        try:
            image_files = sorted(
                [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
        except ValueError:
            rospy.logwarn(f"Could not sort image files numerically in {IMAGE_DIR}. Using alphabetical order.")
            image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])

        for filename in image_files:
            try:
                img_path = os.path.join(IMAGE_DIR, filename)
                images.append(Image.open(img_path))
            except Exception as e:
                rospy.logerr(f"Failed to load image {filename}: {e}")
        rospy.loginfo(f"Successfully loaded {len(images)} images.")
        return images

    def load_scene_graph(self):
        """Loads the merged scene graph from the JSON file."""
        if self.mode < 3:
            return None # Not needed for image-only modes
        try:
            with open(SCENE_GRAPH_PATH, 'r') as f:
                sg_data = json.load(f)
                rospy.loginfo("Successfully loaded scene graph.")
                return json.dumps(sg_data, indent=2) # Return as a formatted string
        except FileNotFoundError:
            rospy.logerr(f"Scene graph file not found at {SCENE_GRAPH_PATH}")
            return None
        except json.JSONDecodeError:
            rospy.logerr(f"Error decoding JSON from {SCENE_GRAPH_PATH}")
            return None

    def generate_prompt(self, question):
        """Constructs the prompt for the Gemini API based on the current mode."""
        base_prompt = f"You are a helpful assistant. Analyze the provided image(s) to answer the following question. The user is a robot, so be precise.\n\nQuestion: \"{question}\""
        
        # Mode-specific instructions
        if self.mode == 1:
            instruction = "Your answer MUST be only a single integer number and nothing else."
        elif self.mode == 2:
            instruction = "First, provide a single integer number as the answer on the first line. On the next line, provide a explanation for your answer."
        elif self.mode == 3:
            instruction = "Use the provided scene graph data to inform your answer. Your answer MUST be only a single integer number and nothing else."
        elif self.mode == 4:
            instruction = "Use the provided scene graph data to inform your answer. First, provide a single integer number as the answer on the first line. On the next line, provide a explanation for your answer."
        else:
            rospy.logwarn(f"Invalid mode: {self.mode}. Defaulting to mode 1.")
            instruction = "Your answer MUST be only a single integer number and nothing else."

        # Combine parts into the final prompt
        full_prompt = [base_prompt, instruction]

        # Add scene graph if the mode requires it
        if self.mode >= 3 and self.scene_graph:
            full_prompt.append("\n\nHere is the scene graph data for context:\n")
            full_prompt.append(self.scene_graph)
        
        # Add images
        full_prompt.extend(self.images)
        
        return full_prompt

    def question_callback(self, msg):
        """Handles incoming questions, generates an answer, and publishes it."""
        question = msg.data
        rospy.loginfo(f"Received question: \"{question}\"")

        if not self.images:
            rospy.logerr("Cannot process question: No images were loaded.")
            return

        prompt_parts = self.generate_prompt(question)
        
        try:
            rospy.loginfo("Sending request to Gemini API...")
            response = self.model.generate_content(prompt_parts)
            answer = response.text
            
            rospy.loginfo(f"Received answer from Gemini: \"{answer}\"")
            self.response_pub.publish(String(data=answer))
            rospy.loginfo("Answer published to /numerical_response.")

        except Exception as e:
            rospy.logerr(f"Failed to get response from Gemini API: {e}")

if __name__ == '__main__':
    try:
        NumericalAnswerGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"An unhandled error occurred: {e}")
