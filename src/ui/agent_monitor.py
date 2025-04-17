import tkinter as tk
from tkinter import ttk
import sv_ttk
from typing import List, Optional
from datetime import datetime
from PIL import ImageGrab

class AgentMonitorUI:
    def __init__(self, title: str = "LLM Agent Thoughts", screen_recorder=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x800")  # Narrower, taller window
        
        # Configure dark theme colors
        self.colors = {
            'bg': '#1e1e1e',        # Dark background
            'text': '#ffffff',       # White text
            'frame': '#2d2d2d',      # Slightly lighter background for frames
            'highlight': '#3d3d3d',  # Highlight color for active elements
            'accent': '#007acc'      # Blue accent color
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg'])
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('default')  # Reset theme to ensure consistent styling
        
        # Configure all frame elements to use dark theme
        self.style.configure('Dark.TFrame', background=self.colors['bg'])
        self.style.configure('Dark.TLabelframe', 
                           background=self.colors['frame'],
                           padding=10,
                           borderwidth=2)
        
        # Configure the label (title) part of the labelframe with larger font
        self.style.configure('Dark.TLabelframe.Label', 
                           background=self.colors['frame'],
                           foreground=self.colors['text'],
                           font=('Arial', 14, 'bold'))  # Increased from 12 to 14
        
        # Additional styling for nested elements
        self.style.map('Dark.TLabelframe', 
                      background=[('active', self.colors['frame'])])
        self.style.map('Dark.TLabelframe.Label', 
                      background=[('active', self.colors['frame'])])
        
        # Configure scrollbar style
        self.style.configure('Dark.Vertical.TScrollbar',
                           background=self.colors['frame'],
                           arrowcolor=self.colors['text'],
                           bordercolor=self.colors['frame'],
                           troughcolor=self.colors['bg'])
        
        self.style.configure('Dark.TButton', 
                           background=self.colors['accent'],
                           foreground=self.colors['text'],
                           padding=5)
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Context History Section
        self.history_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Context History",
            style='Dark.TLabelframe'
        )
        self.history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.history_text = self._create_text_widget(self.history_frame, height=15)
        
        # Last Action Section
        self.action_frame = ttk.LabelFrame(
            self.main_frame,
            text="Last Action",
            style='Dark.TLabelframe'
        )
        self.action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.action_text = self._create_text_widget(self.action_frame, height=4)
        
        # Currently Executing Action Section
        self.executing_frame = ttk.LabelFrame(
            self.main_frame,
            text="Currently Executing Action",
            style='Dark.TLabelframe'
        )
        self.executing_frame.pack(fill=tk.X, pady=(0, 10))
        self.executing_text = self._create_text_widget(self.executing_frame, height=2)
        
        # Steps Counter Section
        self.steps_frame = ttk.LabelFrame(
            self.main_frame,
            text="Steps Taken",
            style='Dark.TLabelframe'
        )
        self.steps_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.steps_text = self._create_text_widget(self.steps_frame, height=1)
        self._add_scrollbar(self.steps_frame, self.steps_text)
        
        # Initialize steps counter
        self.steps_count = 0
        self.update_steps_count(0)  # Initialize display with 0
        
        # Reflection Memory Section
        self.reflection_frame = ttk.LabelFrame(
            self.main_frame,
            text="Reflection Memory",
            style='Dark.TLabelframe'
        )
        self.reflection_frame.pack(fill=tk.BOTH, expand=True)
        
        self.reflection_text = self._create_text_widget(self.reflection_frame, height=10)
        
        
        # Add scrollbars to text widgets
        self._add_scrollbar(self.history_frame, self.history_text)
        self._add_scrollbar(self.action_frame, self.action_text)
        self._add_scrollbar(self.executing_frame, self.executing_text)
        self._add_scrollbar(self.steps_frame, self.steps_text)
        self._add_scrollbar(self.reflection_frame, self.reflection_text)
        
        # Add screen recorder
        self.game_instance = None
        
        # Add recording controls
        self.recording_frame = tk.Frame(
            self.root,
            bg=self.colors['bg']
        )
        self.recording_frame.pack(fill=tk.X, padx=5, pady=5)
        
        sv_ttk.set_theme("dark")
        self.root.update_idletasks()
        self.root.update()

    def _create_text_widget(self, parent, height=10):
        """Create a styled text widget."""
        text_widget = tk.Text(
            parent,
            height=height,
            wrap=tk.WORD,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],  # Cursor color
            selectbackground=self.colors['accent'],
            selectforeground=self.colors['text'],
            font=('Consolas', 10),  # Monospace font for better readability
            padx=5,
            pady=5,
            relief=tk.FLAT,  # Flat appearance
            borderwidth=0
        )
        return text_widget

    def _add_scrollbar(self, parent, text_widget):
        """Add a styled scrollbar to a text widget."""
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=text_widget.yview,
            style='Dark.Vertical.TScrollbar'
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

    def update_context_history(self, history: List[dict]) -> None:
        """Update the context history display."""
        try:
            self.history_text.delete(1.0, tk.END)
            for message in history:
                # Handle Message objects from context_history
                if hasattr(message, 'role') and hasattr(message, 'content'):
                    if isinstance(message.content, list):
                        content = ' '.join([item.get('text', '') for item in message.content 
                                         if item.get('type') == 'text'])
                    else:
                        content = str(message.content)
                    self._insert_formatted_message(self.history_text, message.role, content)
                # Handle dictionary format
                elif isinstance(message, dict):
                    content = message.get('content', '')
                    role = message.get('role', '')
                    self._insert_formatted_message(self.history_text, role, content)
            
            self.history_text.see(tk.END)
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating context history: {e}")

    def _insert_formatted_message(self, text_widget, role, content):
        """Insert a formatted message into the text widget."""
        text_widget.insert(tk.END, f"{role}: ", "role")
        text_widget.insert(tk.END, f"{content}\n", "content")
        # Configure tags for different text styles
        text_widget.tag_configure("role", foreground=self.colors['accent'])
        text_widget.tag_configure("content", foreground=self.colors['text'])

    def update_last_action(self, action: str) -> None:
        """Update the last action display."""
        try:
            self.action_text.delete(1.0, tk.END)
            self.action_text.insert(tk.END, action)
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating last action: {e}")

    def update_executing_action(self, action: str) -> None:
        """Update the currently executing action display."""
        try:
            self.executing_text.delete(1.0, tk.END)
            self.executing_text.insert(tk.END, action)
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating executing action: {e}")

    def update_steps_count(self, count: Optional[int] = None) -> None:
        """Update the steps counter display."""
        try:
            if count is not None:
                self.steps_count = count
            self.steps_text.delete(1.0, tk.END)
            self.steps_text.insert(tk.END, f"Total steps: {self.steps_count}")
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating steps count: {e}")

    def update_reflection_memory(self, memory: str) -> None:
        """Update the reflection memory display."""
        try:
            self.reflection_text.delete(1.0, tk.END)
            self.reflection_text.insert(tk.END, memory)
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating reflection memory: {e}")

    def take_screenshot(self, path: str, name: str) -> None:
        """Take a screenshot of the UI window."""
        try:
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            
            screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            screenshot.save(f"{path}/{name}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")


    def close(self) -> None:
        """Close the UI window."""
        self.root.destroy() 