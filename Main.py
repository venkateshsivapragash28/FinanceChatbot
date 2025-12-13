from dotenv import load_dotenv
from groq import Groq
import os
import yfinance as yf
import json

# -------------------- TOOLS --------------------

def get_stock_price(ticker_symbol: str) -> str:
    try:
        ticker = yf.Ticker(ticker_symbol.upper())
        info = ticker.info

        if "currentPrice" in info and info["currentPrice"] is not None:
            return f"{ticker_symbol.upper()} current price: ${info['currentPrice']}"

        hist = ticker.history(period="1d")
        if not hist.empty:
            return f"{ticker_symbol.upper()} last close: ${hist['Close'].iloc[-1]}"

        return f"No price data found for {ticker_symbol}."

    except Exception as e:
        return f"Error fetching price for {ticker_symbol}: {e}"


def get_market_actives() -> str:
    # yfinance does NOT give a clean "most active" API
    # So we return a placeholder but structured response
    return (
        "Most active stocks typically include large-cap names like:\n"
        "TSLA, AAPL, NVDA, AMZN, MSFT\n"
        "(Note: Live volume ranking requires a paid market data API.)"
    )


def get_market_news(query: str) -> str:
    # Placeholder – no live news API wired yet
    return (
        f"Market news lookup requested for: '{query}'.\n"
        "To enable real news, integrate NewsAPI, Finnhub, or Alpha Vantage."
    )

# -------------------- MAIN APP --------------------

def main():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        return

    client = Groq(api_key=groq_api_key)

    model_name = "llama-3.3-70b-versatile"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g. TSLA, AAPL)"
                        }
                    },
                    "required": ["ticker_symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_actives",
                "description": "Get the most active traded stocks",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_news",
                "description": "Get news or reasons for stock movement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    system_prompt = """
You are a stock market assistant.

STRICT TOOL RULES:

- Use get_stock_price ONLY if the user asks for price, value, or trading price.
- Use get_market_actives ONLY if the user asks for most active or top traded stocks.
- Use get_market_news ONLY if the user asks WHY a stock is moving, news, reasons, or events.

IMPORTANT:
- If the user asks "why", NEVER use get_stock_price.
- Mentioning a company name alone does NOT mean price.
- If no tool applies, answer normally.
"""

    messages = [{"role": "system", "content": system_prompt}]

    tool_executor = {
        "get_stock_price": get_stock_price,
        "get_market_actives": get_market_actives,
        "get_market_news": get_market_news,
    }

    print(f"🤖 Chatbot running on {model_name}")
    print("Type 'exit' to quit")
    print("-" * 40)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # ---------- TOOL CALL ----------
            if msg.tool_calls:
                messages.append(msg)

                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")

                    print(f"🔧 Tool called: {name} {args}")

                    result = tool_executor[name](**args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": result
                    })

                final = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )

                output = final.choices[0].message.content
            else:
                output = msg.content

            print(f"Bot: {output}")
            messages.append({"role": "assistant", "content": output})

        except Exception as e:
            print(f"❌ Error: {e}")
            messages.pop()

if __name__ == "__main__":
    main()
