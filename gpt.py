from openai import OpenAI
class GPT():
  def __init__(self, model = 'gpt-4o-mini'):

    self.client = OpenAI()
    self.model = model

  def get_answer(self, prompt):
    response =  self.client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt
        }
        ]
    )
    return response.choices[0].message.content.strip()