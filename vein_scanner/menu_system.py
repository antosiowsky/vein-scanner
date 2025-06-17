# menu_system.py

import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import hardware_config as hw
import logging
from gpiozero import Button

class MenuSystem:
    def __init__(self, param_setter, start_scan_callback, stop_scan_callback, shutdown_callback):
        # ... (__init__ setup is unchanged)
        self.param_setter = param_setter
        self.start_scan_callback = start_scan_callback
        self.stop_scan_callback = stop_scan_callback
        self.shutdown_callback = shutdown_callback
        self.oled = adafruit_ssd1306.SSD1306_I2C(hw.OLED_WIDTH, hw.OLED_HEIGHT, busio.I2C(board.SCL, board.SDA))
        self.oled.fill(0); self.oled.show()
        self.font = ImageFont.load_default()
        try: self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except IOError: self.small_font = self.font
        self.menu_stack = ['main']
        self.current_selection = 0
        self.edit_mode = False
        self.scan_mode_active = False

        # --- UPDATED MENU DEFINITIONS ---
        self.menus = {
            'main': {
                'title': 'Main Menu',
                'options': [
                    ('[Scan Action]', self._action_toggle_scan),
                    ('Settings', lambda: self._navigate_to('settings')),
                    ('Shutdown', self._action_shutdown)
                ]
            },
            'settings': {
                'title': 'Settings',
                'options': [
                    # NOTE: Gain, Clip Limit, Tile Size are no longer in the menu
                    # but will still be loaded from the config file.
                    ('Exposure', lambda: self._enter_edit_mode('exposure', 100, 'Exposure')),
                    ('For Value', lambda: self._enter_edit_mode('clahe_for_value', 1, 'For Value')), # <-- ADDED
                    ('Invert Colors', self._action_toggle_invert), # <-- ADDED
                    ('Back', self.back)
                ]
            }
        }

        # Button setup is unchanged
        self.btn_up = Button(hw.BTN_UP_PIN, pull_up=True, bounce_time=0.05); self.btn_up.when_pressed = self.handle_up
        self.btn_down = Button(hw.BTN_DOWN_PIN, pull_up=True, bounce_time=0.05); self.btn_down.when_pressed = self.handle_down
        self.btn_select = Button(hw.BTN_SELECT_PIN, pull_up=True, bounce_time=0.05); self.btn_select.when_pressed = self.handle_select
        self.btn_back = Button(hw.BTN_BACK_PIN, pull_up=True, bounce_time=0.05); self.btn_back.when_pressed = self.back
        
        self.display_menu()

    def display_menu(self):
        image = Image.new("1", (self.oled.width, self.oled.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.oled.width, self.oled.height), outline=0, fill=0)

        if self.edit_mode:
            value_format = "{:.1f}" if isinstance(self.edit_param_value, float) else "{:d}"
            draw.text((0, 0), f"Edit {self.edit_param_friendly_name}", font=self.font, fill=255)
            draw.text((0, 20), f"Value: {value_format.format(self.edit_param_value)}", font=self.font, fill=255)
            draw.text((0, 40), "OK to Save, Back to Cancel", font=self.small_font, fill=255)
        else:
            current_menu_key = self.menu_stack[-1]
            menu = self.menus[current_menu_key]
            draw.text((0, 0), menu['title'], font=self.font, fill=255)
            y = 15
            
            for i, (text_placeholder, _) in enumerate(menu['options']):
                prefix = "> " if i == self.current_selection else "  "
                display_text = f"{prefix}{text_placeholder}"
                
                if text_placeholder == '[Scan Action]':
                    display_text = f"{prefix}{'Stop Scan' if self.scan_mode_active else 'Start Scan'}"
                elif current_menu_key == 'settings':
                    # --- UPDATED DISPLAY LOGIC ---
                    if text_placeholder == 'Exposure':
                        display_text += f": {self.param_setter.exposure}"
                    elif text_placeholder == 'For Value':
                        display_text += f": {self.param_setter.clahe_for_value}"
                    elif text_placeholder == 'Invert Colors':
                        state = "On" if self.param_setter.invert_colors else "Off"
                        display_text += f": {state}"
                
                draw.text((0, y), display_text, font=self.font, fill=255)
                y += 12

        self.oled.image(image); self.oled.show()

    def handle_up(self, button=None):
        if self.edit_mode:
            attr = self.edit_param_value_attr
            new_val = round(self.edit_param_value + self.edit_step, 2)
            if attr == 'exposure': # Max exposure is now 500
                new_val = min(500, new_val)
            self.edit_param_value = new_val
        else:
            self.current_selection = (self.current_selection - 1) % len(self.menus[self.menu_stack[-1]]['options'])
        self.display_menu()

    def handle_down(self, button=None):
        if self.edit_mode:
            attr = self.edit_param_value_attr
            new_val = round(self.edit_param_value - self.edit_step, 2)
            if attr == 'exposure': # Min exposure is 1
                new_val = max(1, new_val)
            elif attr == 'clahe_for_value': # Min for value is 1
                new_val = max(1, new_val)
            self.edit_param_value = new_val
        else:
            self.current_selection = (self.current_selection + 1) % len(self.menus[self.menu_stack[-1]]['options'])
        self.display_menu()

    def handle_select(self, button=None):
        if self.edit_mode: self._save_edit()
        else:
            _, action = self.menus[self.menu_stack[-1]]['options'][self.current_selection]
            action()

    # --- NEW ACTION METHOD FOR INVERT TOGGLE ---
    def _action_toggle_invert(self):
        """Toggles the 'invert_colors' boolean parameter and saves it."""
        self.param_setter.invert_colors = not self.param_setter.invert_colors
        self.param_setter.save_params_to_file()
        logging.info(f"Set Invert Colors to: {self.param_setter.invert_colors}")
        self.display_menu() # Redraw to show the new state

    def _save_edit(self):
        # All editable params are now integers
        setattr(self.param_setter, self.edit_param_value_attr, int(round(self.edit_param_value)))
        self.param_setter.save_params_to_file()
        self.edit_mode = False
        self.display_menu()
        
    # --- The rest of the file is unchanged ---
    def back(self, button=None):
        if self.edit_mode: self.edit_mode = False
        elif len(self.menu_stack) > 1: self.menu_stack.pop(); self.current_selection = 0
        self.display_menu()
    def _action_toggle_scan(self):
        self.scan_mode_active = not self.scan_mode_active
        if self.scan_mode_active: self.start_scan_callback()
        else: self.stop_scan_callback()
        self.display_menu()
    def _navigate_to(self, menu_key):
        self.menu_stack.append(menu_key); self.current_selection = 0; self.display_menu()
    def _enter_edit_mode(self, attr_name, step, friendly_name):
        self.edit_mode = True; self.edit_param_friendly_name = friendly_name
        self.edit_param_value_attr = attr_name; self.edit_param_value = getattr(self.param_setter, attr_name)
        self.edit_step = step; self.display_menu()
    def _action_shutdown(self):
        self.oled.fill(0); draw = ImageDraw.Draw(self.oled.image)
        draw.text((0, 20), "Shutting Down...", font=self.font, fill=255)
        self.oled.show(); self.shutdown_callback()
    def cleanup(self):
        self.btn_up.close(); self.btn_down.close(); self.btn_select.close(); self.btn_back.close()
        self.oled.fill(0); self.oled.show()