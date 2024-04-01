# TinyAgent

在`ChatGPT`横空出世，夺走`Bert`的桂冠之后，大模型愈发的火热，国内各种模型层出不穷，史称“百模大战”。大模型的能力是毋庸置疑的，但大模型在一些实时的问题上，或是某些专有领域的问题上，可能会显得有些力不从心。因此，我们需要一些工具来为大模型赋能，给大模型一个抓手，让大模型和现实世界发生的事情对齐颗粒度，这样我们就获得了一个更好的用的大模型。

这里不要葱姜蒜同学基于`React`的方式，制作了一个最小的`Agent`结构（其实更多的是调用工具），暑假的时候会尝试将React结构修改为SOP结构。

一步一步手写`Agent`，可能让我对`Agent`的构成和运作更加的了解。以下是`React`论文中一些小例子。

> 论文：***[ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)***

<div style="display: flex; justify-content: center;">
    <img src="./images/React.png" style="width: 100%;">
</div>

## 实现细节

### Step 1: 构造大模型

我们需要一个大模型，这里我们使用`InternLM2`作为我们的大模型。`InternLM2`是一个基于`Decoder-Only`的对话大模型，我们可以使用`transformers`库来加载`InternLM2`。

首先，还是先创建一个`BaseModel`类，这个类是一个抽象类，我们可以在这个类中定义一些基本的方法，比如`chat`方法和`load_model`方法。方便以后扩展使用其他模型。

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass
```

接着，我们创建一个`InternLM2`类，这个类继承自`BaseModel`类，我们在这个类中实现`chat`方法和`load_model`方法。就和正常加载`InternLM2`模型一样，来做一个简单的加载和返回即可。

```python
class InternLM2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
        return response, history
```

### Step 2: 构造工具

我们在`tools.py`文件中，构造一些工具，比如`Google搜索`。我们在这个文件中，构造一个`Tools`类，这个类中包含了一些工具的描述信息和具体实现。我们可以在这个类中，添加一些工具的描述信息和具体实现。

- 首先要在 `tools` 中添加工具的描述信息
- 然后在 `tools` 中添加工具的具体实现

> *使用Google搜索功能的话需要去`serper`官网申请一下`token`: https://serper.dev/dashboard*

```python
class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def google_search(self, search_query: str):
        pass
```

### Step 3: 构造Agent

我们在`Agent`类中，构造一个`Agent`，这个`Agent`是一个`React`的`Agent`，我们在这个`Agent`中，实现了`chat`方法，这个方法是一个对话方法，我们在这个方法中，调用`InternLM2`模型，然后根据`React`的`Agent`的逻辑，来调用`Tools`中的工具。

首先我们要构造`system_prompt`, 这个是系统的提示，我们可以在这个提示中，添加一些系统的提示信息，比如`ReAct`形式的`prompt`。

```python
def build_system_input(self):
    tool_descs, tool_names = [], []
    for tool in self.tool.toolConfig:
        tool_descs.append(TOOL_DESC.format(**tool))
        tool_names.append(tool['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)
    sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
    return sys_prompt
```

OK, 如果顺利的话，运行出来的示例应该是这样的：
    
```
Answer the following questions as best you can. You have access to the following tools:

google_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{'name': 'search_query', 'description': '搜索关键词或短语', 'required': True, 'schema': {'type': 'string'}}] Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [google_search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
```

这个`system_prompt`告诉了大模型，它可以调用哪些工具，以什么样的方式输出，以及工具的描述信息和工具应该接受什么样的参数。

> 目前只是实现了一个简单的`Google搜索`工具，后续会添加更多的关于地理信息系统分析的工具，没错，我是一个地理信息系统的学生。

关于Agent的具体结构可以在Agent.py中查看。这里就简单说一下，Agent的结构是一个`React`的结构，提供一个`system_prompt`，使得大模型知道自己可以调用那些工具，并以什么样的格式输出。

每次用户的提问，如果需要调用工具的话，都会进行两次的大模型调用，第一次是用户的提问获得工具调用的参数有，第二次是用户的提问+工具的调用。这样就可以实现一个`React`的结构。

### Step 4: 运行Agent

在这个案例中，使用了`InternLM2-chat-7B`模型， 如果你想要`Agent`运行的更加稳定，可以使用它的`big cup`版本`InternLM2-20b-chat`，这样可以提高`Agent`的稳定性。

```python
from Agent import Agent


agent = Agent('/root/share/model_repos/internlm2-chat-20b')

response, _ = agent.text_completion(text='你好', history=[])
print(response)

# Thought: 你好，请问有什么我可以帮助你的吗？
# Action: google_search
# Action Input: {'search_query': '你好'}
# Observation:Many translated example sentences containing "你好" – English-Chinese dictionary and search engine for English translations.
# Final Answer: 你好，请问有什么我可以帮助你的吗？ 

response, _ = agent.text_completion(text='特朗普哪一年出生的？', history=_)
print(response)

# Thought: 根据我的搜索结果，唐纳德·特朗普的出生年份是1946年。
# Final Answer: 1946 

response, _ = agent.text_completion(text='周杰伦是谁？', history=_)
print(response)

# Thought: 根据我的搜索结果，周杰伦是一位台湾的创作男歌手、钢琴家和词曲作家。他的首张专辑《杰倫》于2000年推出，他的音乐遍及亚太区和西方国家。
# Final Answer: 周杰伦是一位台湾的创作男歌手、钢琴家和词曲作家。他的首张专辑《杰倫》于2000年推出，他的音乐遍及亚太区和西方国家。 
```

## 论文参考

- [ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)