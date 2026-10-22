import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app
import database
import ai_service

client = TestClient(app.app)


class TestGeminiChatbotIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure fresh database tables
        database.init_db()

    def test_01_give_me_a_diet_plan(self):
        """Test 'Give me a diet plan.' with mocked Gemini response."""
        mock_response = MagicMock()
        mock_response.text = "Here is a balanced daily meal plan with oats for breakfast, dal and roti for lunch, and vegetable soup for dinner."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "Give me a diet plan.",
                "user_id": "test_user_diet"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("breakfast", data["message"].lower())
            self.assertTrue(len(data["medical_references"]) > 0)
            self.assertTrue(any("nutrition" in r["title"].lower() or "eating" in r["title"].lower() for r in data["medical_references"]))

    def test_02_how_can_i_reduce_my_bmi(self):
        """Test 'How can I reduce my BMI?'."""
        mock_response = MagicMock()
        mock_response.text = "To reduce your BMI sustainably, focus on a calorie-conscious diet, 30 minutes of daily exercise, and consistent sleep."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "How can I reduce my BMI?",
                "user_id": "test_user_bmi"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("bmi", data["message"].lower())

    def test_03_what_is_bmi(self):
        """Test 'What is BMI?'."""
        mock_response = MagicMock()
        mock_response.text = "BMI stands for Body Mass Index. It is calculated by dividing weight in kilograms by height in meters squared."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "What is BMI?",
                "user_id": "test_user_bmi2"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("body mass index", data["message"].lower())

    def test_04_explain_predicted_premium_and_grounding(self):
        """Test 'Explain my predicted premium.' uses verified prediction from backend."""
        # Insert a verified prediction for the user
        test_pred = {
            "risk": "Low",
            "cluster": 2,
            "risk_model": "LightGBM Regressor",
            "risk_aware_prediction": 8130.50,
            "global_prediction": 12500.00,
            "top_shap_factors": [{"feature": "bmi", "direction": "Decreases Claim", "impact": -1200.0}]
        }
        database.save_prediction("user_premium_test", test_pred, {"age": 25, "bmi": 21.0})

        captured_config = []

        def mock_generate_content(model, contents, config):
            captured_config.append(config)
            resp = MagicMock()
            resp.text = "Your verified annual premium prediction is ₹8,130.50 under the Low Risk tier because of your healthy BMI."
            return resp

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = mock_generate_content
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "Explain my predicted premium.",
                "user_id": "user_premium_test"
            })
            self.assertEqual(res.status_code, 200)
            # Verify system instruction contains the verified backend prediction data
            self.assertTrue(len(captured_config) > 0)
            sys_inst = captured_config[0].system_instruction
            self.assertIn("8,130.50", sys_inst)
            self.assertIn("Low Risk", sys_inst)

    def test_05_unrelated_questions_do_not_receive_insurance_context(self):
        """Verify that unrelated questions (e.g. diet) do NOT automatically receive insurance prediction context."""
        # Insert a verified prediction for user
        test_pred = {
            "risk": "High",
            "risk_aware_prediction": 35000.00,
            "global_prediction": 15000.00,
            "top_shap_factors": []
        }
        database.save_prediction("user_diet_isolation", test_pred, {"age": 55})

        captured_config = []

        def mock_generate_content(model, contents, config):
            captured_config.append(config)
            resp = MagicMock()
            resp.text = "Here is a simple diet plan: vegetable oats for breakfast, rice and dal for lunch."
            return resp

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = mock_generate_content
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "Give me a diet plan.",
                "user_id": "user_diet_isolation"
            })
            self.assertEqual(res.status_code, 200)
            sys_inst = captured_config[0].system_instruction
            # Must NOT include 35,000 or insurance context for diet request
            self.assertNotIn("35,000", sys_inst)
            self.assertNotIn("VERIFIED USER INSURANCE PREDICTION", sys_inst)

    def test_06_what_does_low_risk_mean(self):
        """Test 'What does Low Risk mean?'."""
        mock_response = MagicMock()
        mock_response.text = "Low Risk means your statistical profile has favorable indicators like normal blood pressure and non-smoking habits."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "What does Low Risk mean?",
                "user_id": "test_user_risk"
            })
            self.assertEqual(res.status_code, 200)

    def test_07_how_can_i_improve_my_sleep(self):
        """Test 'How can I improve my sleep?'."""
        mock_response = MagicMock()
        mock_response.text = "To improve your sleep, go to bed at a consistent time, put screens away 45 minutes before sleep, and avoid caffeine in the afternoon."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "How can I improve my sleep?",
                "user_id": "test_user_sleep"
            })
            self.assertEqual(res.status_code, 200)

    def test_08_vegetarian_weight_management_meal_plan(self):
        """Test 'Give me a vegetarian weight-management meal plan.'."""
        mock_response = MagicMock()
        mock_response.text = "Here is a balanced vegetarian meal plan featuring poha, paneer salad, rajma with brown rice, and roasted makhana."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "Give me a vegetarian weight-management meal plan.",
                "user_id": "test_user_veg"
            })
            self.assertEqual(res.status_code, 200)

    def test_09_height_weight_bmi_calculation(self):
        """Test providing height and weight, then requesting BMI calculation."""
        mock_response = MagicMock()
        mock_response.text = "For a height of 170 cm and weight of 80 kg, your BMI is approximately 27.7, which falls into the overweight category."

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "I am 170 cm and 80 kg. Calculate my BMI.",
                "user_id": "test_user_bmi_calc"
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn("27.7", res.json()["message"])

    def test_10_follow_up_remembers_previous_turn(self):
        """Test multi-turn context retention across conversation turns."""
        conv = database.create_conversation("user_turn_test", "Diet Conversation")
        conv_id = conv["id"]

        captured_contents = []

        def mock_generate(model, contents, config):
            captured_contents.append(contents)
            resp = MagicMock()
            resp.text = "Since you are vegetarian and 170 cm, here is a customized plant-based meal plan."
            return resp

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = mock_generate
            mock_client_cls.return_value = mock_client

            # Turn 1
            res1 = client.post("/api/chat", json={
                "message": "I am vegetarian and 170 cm.",
                "conversation_id": conv_id,
                "user_id": "user_turn_test"
            })
            self.assertEqual(res1.status_code, 200)

            # Turn 2
            res2 = client.post("/api/chat", json={
                "message": "Give me a meal plan.",
                "conversation_id": conv_id,
                "user_id": "user_turn_test"
            })
            self.assertEqual(res2.status_code, 200)

            # Verify Turn 2 passed previous turn into Gemini contents
            turn2_contents = captured_contents[1]
            all_texts = []
            for c in turn2_contents:
                for p in c.parts:
                    all_texts.append(p.text)
            self.assertTrue(any("vegetarian and 170 cm" in t for t in all_texts))
            self.assertTrue(any("Give me a meal plan." in t for t in all_texts))

    def test_11_latest_user_message_passed_exactly_once(self):
        """Verify that the latest user message is passed exactly once to Gemini."""
        conv = database.create_conversation("user_dup_test", "Deduplication Test")
        conv_id = conv["id"]

        captured_contents = []

        def mock_generate(model, contents, config):
            captured_contents.append(contents)
            resp = MagicMock()
            resp.text = "Answer"
            return resp

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = mock_generate
            mock_client_cls.return_value = mock_client

            client.post("/api/chat", json={
                "message": "UniqueQuery12345",
                "conversation_id": conv_id,
                "user_id": "user_dup_test"
            })

            contents = captured_contents[0]
            # Count occurrences of UniqueQuery12345 in all parts
            count = sum(1 for c in contents for p in c.parts if "UniqueQuery12345" in p.text)
            self.assertEqual(count, 1)

    def test_12_separate_conversations_isolation(self):
        """Verify that separate conversations remain completely isolated."""
        conv1 = database.create_conversation("user_iso", "Conv 1")
        conv2 = database.create_conversation("user_iso", "Conv 2")

        database.add_message(conv1["id"], role="user", content="I have a cat")
        database.add_message(conv2["id"], role="user", content="I have a dog")

        msgs1 = database.get_messages(conv1["id"])
        msgs2 = database.get_messages(conv2["id"])

        self.assertEqual(len(msgs1), 1)
        self.assertEqual(msgs1[0]["content"], "I have a cat")
        self.assertEqual(len(msgs2), 1)
        self.assertEqual(msgs2[0]["content"], "I have a dog")

    def test_13_unauthorized_access_denied(self):
        """Verify that attempting to access or send to another user's conversation is denied (403)."""
        conv = database.create_conversation("legit_user", "Private Chat")
        conv_id = conv["id"]

        # Other user attempts to chat into legit_user's conversation
        res = client.post("/api/chat", json={
            "message": "Spying message",
            "conversation_id": conv_id,
            "user_id": "attacker_user"
        })
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access denied", res.json()["detail"])

        # Other user attempts to fetch messages
        res_get = client.get(f"/api/conversations/{conv_id}/messages?user_id=attacker_user")
        self.assertEqual(res_get.status_code, 403)

        # Other user attempts to delete conversation
        res_del = client.delete(f"/api/conversations/{conv_id}?user_id=attacker_user")
        self.assertEqual(res_del.status_code, 403)

    def test_14_gemini_api_failure_handling(self):
        """Verify that Gemini API failure returns clear error without saving a fake AI response."""
        conv = database.create_conversation("user_fail_test", "Failure Test")
        conv_id = conv["id"]

        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("Simulated Gemini quota exceeded")
            mock_client_cls.return_value = mock_client

            res = client.post("/api/chat", json={
                "message": "Should fail",
                "conversation_id": conv_id,
                "user_id": "user_fail_test"
            })
            self.assertEqual(res.status_code, 502)
            self.assertIn("Gemini AI service error", res.json()["detail"])

            # Verify NO assistant message was saved to the database
            messages = database.get_messages(conv_id)
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            self.assertEqual(len(assistant_messages), 0)

    def test_15_missing_api_key_configuration(self):
        """Verify that missing API key returns 503 with clear explanation."""
        with patch("ai_service.get_gemini_api_key", return_value=""):
            res = client.post("/api/chat", json={
                "message": "Hello without key",
                "user_id": "user_no_key"
            })
            self.assertEqual(res.status_code, 503)
            self.assertIn("Gemini API key is not configured", res.json()["detail"])

    def test_16_idempotency_on_retry(self):
        """Verify that retrying after an error does not produce duplicate user messages in DB."""
        conv = database.create_conversation("user_retry_test", "Retry Test")
        conv_id = conv["id"]

        # First attempt: Gemini fails
        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("Network timeout")
            mock_client_cls.return_value = mock_client

            res1 = client.post("/api/chat", json={
                "message": "Retry query",
                "conversation_id": conv_id,
                "user_id": "user_retry_test"
            })
            self.assertEqual(res1.status_code, 502)

        # Check DB: user message was recorded, no assistant message
        msgs1 = database.get_messages(conv_id)
        self.assertEqual(len(msgs1), 1)
        self.assertEqual(msgs1[0]["role"], "user")

        # Second attempt: Gemini succeeds on retry
        mock_resp = MagicMock()
        mock_resp.text = "Success on retry"
        with patch("ai_service.get_gemini_api_key", return_value="AIzaSyFakeKeyForTest123"), \
             patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            res2 = client.post("/api/chat", json={
                "message": "Retry query",
                "conversation_id": conv_id,
                "user_id": "user_retry_test"
            })
            self.assertEqual(res2.status_code, 200)

        # Check DB: user message is NOT duplicated (only 1 user message, 1 assistant message)
        msgs2 = database.get_messages(conv_id)
        self.assertEqual(len(msgs2), 2)
        self.assertEqual(msgs2[0]["role"], "user")
        self.assertEqual(msgs2[1]["role"], "assistant")
        self.assertEqual(msgs2[1]["content"], "Success on retry")


if __name__ == "__main__":
    unittest.main()
