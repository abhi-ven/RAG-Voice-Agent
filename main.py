#Flet UI Main 

import flet as ft
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from queue import Queue
import numpy as np
import time
import threading
from agent import agent
from record import record, transcribe, texttospeech
from streaming import stream_response

def main(page: ft.Page):
    messages = []

    score = 0
    questions_asked = 0

    def safe_update(control=None):
        try:
            target = control if control else page
            page.run_thread(lambda: target.update())
        except Exception:
            pass

    page.title = "Voice Agent"
    page.bgcolor = ft.Colors.BLUE_100
    page.padding = 16

    chat_list = ft.ListView(expand=1, spacing=8, auto_scroll=True, padding=4)

    def add_user_bubble(text: str):
        bubble = ft.Container(
            content=ft.Text(text, size=13),
            bgcolor=ft.Colors.WHITE,
            border_radius=ft.BorderRadius(top_left=16, top_right=4, bottom_left=16, bottom_right=16),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            width=300,
            alignment=ft.alignment.Alignment(1, 0),
        )
        chat_list.controls.append(ft.Row([bubble], alignment=ft.MainAxisAlignment.END))
        safe_update(chat_list)

    def add_agent_bubble() -> ft.Text:
        agent_text = ft.Text(
            "",
            selectable=True,
            size=13,
        )
        bubble = ft.Container(
            content=agent_text,
            bgcolor=ft.Colors.GREY_200,
            border_radius=ft.BorderRadius(top_left=4, top_right=16, bottom_left=16, bottom_right=16),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            width=300,
        )
        chat_list.controls.append(ft.Row([bubble], alignment=ft.MainAxisAlignment.START))
        safe_update(chat_list)
        return agent_text

    def set_status(text: str, color: str):
        status.value = text
        status.color = color
        safe_update(status)

    def run_pipeline():
        #here
        nonlocal score, questions_asked
        set_status("Transcribing...", ft.Colors.ORANGE)
        chunks = []
        while not dataqueue.empty():
            try:
                chunks.append(dataqueue.get_nowait())
            except Exception:
                break
        audio_data = b"".join(chunks)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        userinput = transcribe(audio_array)

        if "stop now" in userinput.lower():
            page.run_task(page.window.close)  
            return

        if not userinput:
            set_status("Try again.", ft.Colors.RED)
            time.sleep(1.5)
            set_status("Ready", ft.Colors.GREEN)
            micbutton.disabled = False
            safe_update(micbutton)
            return

        add_user_bubble(userinput)

        if questions_asked >= 5:

            if score / questions_asked < 0.5:
                messages.append(SystemMessage(
                    content="Ask easier questions."
                ))

            elif score / questions_asked > 0.8:
                messages.append(SystemMessage(
                    content="Ask more difficult questions."
                ))
        messages.append(HumanMessage(content=userinput))

        set_status("Thinking...", ft.Colors.ORANGE)
        agent_text_widget = add_agent_bubble()

        last_update = [0]

        def on_token(full_so_far: str):
            agent_text_widget.value = full_so_far
            now = time.time()
            if now - last_update[0] > 0.05:  # update at most every 50ms
                last_update[0] = now
                safe_update(agent_text_widget)

        try:
            full_response, new_messages = stream_response(agent, messages, on_token=on_token)
            questions_asked += 1

            response_lower = full_response.strip().lower()

            if response_lower.startswith("correct"):
                score += 1

            score_text.value = f"Score: {score}/{questions_asked}"
            safe_update(score_text)

            # Final update to make sure the complete response is shown
            agent_text_widget.value = full_response
            safe_update(agent_text_widget)
            messages.clear()
            messages.extend(new_messages)
        except Exception as e:
            agent_text_widget.value = f"Error: {e}"
            safe_update(agent_text_widget)

        set_status("Ready", ft.Colors.GREEN)
        micbutton.disabled = False
        safe_update(micbutton)

    recording = False
    record_thread = None

    def toggle(e):
        nonlocal recording, record_thread

        if not recording:
            recording = True
            stop.clear()

            while not dataqueue.empty():
                try: 
                    dataqueue.get_nowait()
                except: 
                    break

            record_thread = threading.Thread(
                target=record, args=(stop, dataqueue), daemon=True
            )
            record_thread.start()

            micbutton.content = ft.Text(" Stop", color=ft.Colors.WHITE)
            micbutton.bgcolor = ft.Colors.RED_400
            set_status("Recording...", ft.Colors.RED)

        else:
            recording = False
            stop.set()

            micbutton.content = ft.Text(" Speak", color=ft.Colors.WHITE)
            micbutton.bgcolor = ft.Colors.BLUE_200
            micbutton.disabled = True
            safe_update(micbutton)

            def wait_then_run():
                if record_thread:
                    record_thread.join()
                run_pipeline()

            threading.Thread(target=wait_then_run, daemon=True).start()

        safe_update(micbutton)

    stop = threading.Event()
    dataqueue = Queue()

    status = ft.Text("Ready", color=ft.Colors.GREEN, size=13, weight=ft.FontWeight.W_500)
    score_text = ft.Text(
        "Score: 0/0",
        color=ft.Colors.BLACK,
        size=13,
        weight=ft.FontWeight.W_500,
)

    micbutton = ft.Button(
        content=ft.Text(" Speak", color=ft.Colors.BLACK),
        on_click=toggle,
        bgcolor=ft.Colors.BLUE_300,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=24),
            padding=ft.Padding(left=24, top=14, right=24, bottom=14),
        ),
    )

    page.add(
        ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Quiz Agent", size=16, weight=ft.FontWeight.W_700),
                        score_text,
                        status,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Divider(height=1, color=ft.Colors.BLACK),
                ft.Container(
                    content=chat_list,
                    expand=True,
                    bgcolor=ft.Colors.WHITE,
                    border_radius=12,
                    padding=12,
                    border=ft.Border(
                        left=ft.BorderSide(1, ft.Colors.BLACK),
                        right=ft.BorderSide(1, ft.Colors.BLACK),
                        top=ft.BorderSide(1,ft.Colors.BLACK),
                        bottom=ft.BorderSide(1, ft.Colors.BLACK)
                    ),
                ),
                ft.Row(
                    controls=[
                        micbutton,
                        ft.Text(
                            "Press Speak to start, Stop when done.",
                            color=ft.Colors.GREY_500,
                            size=12,
                        ),
                    ],
                    spacing=16,
                ),
            ],
            expand=True,
            spacing=12,
        )
    )

ft.run(main)