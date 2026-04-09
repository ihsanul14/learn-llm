import './index.css';
import { createSignal, For, Show } from "solid-js";
import { marked } from 'marked';
import { createMemo } from 'solid-js';
import DOMPurify from 'dompurify';

type Message = { role: "user" | "assistant"; content: string };

function App() {
  const [messages, setMessages] = createSignal<Message[]>([]);
  const [input, setInput] = createSignal("");
  const [loading, setLoading] = createSignal(false);
  const [sidebarOpen, setSidebarOpen] = createSignal(true);

  function SafeMarkdown(props: { content: string }) {
    const rendered = createMemo(() => {
      const rawHtml = marked.parse(props.content) as string;
      return DOMPurify.sanitize(rawHtml);
    });
  
    return <div class="markdown-content" innerHTML={rendered()} />;
  }

  const sendMessage = async () => {
    if (!input().trim() || loading()) return;

    const userMsg: Message = { role: "user", content: input() };
    setMessages([...messages(), userMsg]);
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "assistant", content: "Thinking..." }]);

    try {
      const response = await fetch("http://localhost:30001/api/chat", {
        method: "POST",
        body: JSON.stringify({
          model: "llama3.1",
          messages: [...messages().slice(0, -1), userMsg],
          stream: true,
        }),
      });

      const reader = response.body?.getReader();
      if (!reader) return;

      let assistantText = "";
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        
        for (const line of lines) {
          if (!line.trim()) continue;
          const json = JSON.parse(line);
          if (json.message?.content) {
            assistantText += json.message.content;
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = { role: "assistant", content: assistantText };
              return next;
            });
          }
        }
      }
    } catch (err) {
      console.error("Ollama Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="flex h-screen w-full bg-[#212121] text-gray-100 font-sans overflow-hidden">
      <aside 
        class={`bg-[#171717] transition-all duration-300 flex flex-col border-r border-white/10 ${sidebarOpen() ? "w-64" : "w-0"}`}
      >
        <div class="p-4 flex flex-col h-full">
          <button class="flex items-center gap-3 px-3 py-2 border border-white/20 rounded-md hover:bg-white/5 transition mb-4">
            <span class="text-xl">+</span>
            <span class="text-sm font-medium">New Chat</span>
          </button>
          
          <div class="flex-1 overflow-y-auto space-y-2">
            <p class="text-xs text-gray-500 font-bold px-3 py-2 uppercase tracking-wider">Yesterday</p>
            <div class="px-3 py-2 text-sm truncate rounded-md hover:bg-[#2c2c2c] cursor-pointer text-gray-300">
              How to learn LLM Engineering
            </div>
            <div class="px-3 py-2 text-sm truncate rounded-md hover:bg-[#2c2c2c] cursor-pointer text-gray-300">
              SolidJS vs React in 2026
            </div>
          </div>

          <div class="border-t border-white/10 pt-4">
            <div class="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-[#2c2c2c] cursor-pointer">
              <div class="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-xs">U</div>
              <span class="text-sm font-medium text-gray-200">User Profile</span>
            </div>
          </div>
        </div>
      </aside>

      <main class="flex-1 flex flex-col relative h-full">
        <header class="p-4 flex items-center justify-between bg-[#212121]/80 backdrop-blur-md sticky top-0 z-10">
          <button onClick={() => setSidebarOpen(!sidebarOpen())} class="p-2 hover:bg-white/10 rounded-md transition text-gray-400">
             ☰
          </button>
          <div class="font-semibold text-gray-200">ChatGPT Clone <span class="text-xs text-gray-500 ml-2">Llama 3.2</span></div>
          <div class="w-8"></div>
        </header>
        <div class="flex-1 overflow-y-auto px-4 md:px-0">
          <Show when={messages().length === 0}>
             <div class="h-full flex flex-col items-center justify-center space-y-4 opacity-50">
                <div class="text-4xl font-bold">Ollama GPT</div>
                <p>Start a conversation with your local AI.</p>
             </div>
          </Show>

          <For each={messages()}>
            {(msg) => (
              <div class={`w-full py-8 border-b border-black/10 flex justify-center ${msg.role === 'assistant' ? 'bg-[#2f2f2f]' : 'bg-[#212121]'}`}>
                <div class="max-w-3xl w-full flex gap-6 px-4">
                  <div class={`w-8 h-8 rounded-sm flex items-center justify-center text-white shrink-0 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-emerald-600'}`}>
                    {msg.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div class="flex-1 prose prose-invert leading-relaxed whitespace-pre-wrap" >
                    <SafeMarkdown content={msg.content} />
                  </div>
                </div>
              </div>
            )}
          </For>
          <div class="h-48"></div>
        </div>
        <div class="absolute bottom-0 left-0 w-full bg-linear-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-8 flex justify-center">
          <div class="max-w-3xl w-full px-4 relative">
            <div class="relative flex items-center bg-[#2f2f2f] rounded-xl border border-white/10 shadow-2xl">
              <textarea 
                rows="1"
                class="w-full bg-transparent p-4 pr-16 focus:outline-none resize-none overflow-hidden max-h-52 text-gray-100 placeholder-gray-500"
                placeholder="Message Ollama GPT..."
                value={input()}
                onInput={(e) => setInput(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />
              <button 
                onClick={sendMessage}
                disabled={loading() || !input().trim()}
                class="absolute right-3 bottom-3 p-2 bg-white text-black rounded-lg disabled:bg-gray-600 disabled:text-gray-400 hover:bg-gray-200 transition"
              >
                ↑
              </button>
            </div>
            <p class="text-[10px] text-gray-500 mt-3 text-center uppercase tracking-widest">
              Ollama Engineering Prototype 2026
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;