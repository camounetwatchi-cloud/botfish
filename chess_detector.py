import chess
import chess.engine
import pyautogui
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2
import subprocess
import platform

class ChessComDetector:
    def __init__(self, stockfish_path):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.board_position = None
        
    def start_engine(self):
        """Démarre le moteur Stockfish"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print("✓ Moteur Stockfish démarré")
            return True
        except Exception as e:
            print(f"✗ Erreur lors du démarrage de Stockfish: {e}")
            print("\nATTENTION: Vérifiez que:")
            print("1. Stockfish est téléchargé depuis: https://stockfishchess.org/download/")
            print("2. Le fichier stockfish.exe est dans le bon dossier")
            print("3. Le chemin dans le code est correct")
            return False
    
    def find_chessboard(self):
        """Détecte la position de l'échiquier Chess.com à l'écran"""
        print("\n🔍 Recherche de l'échiquier Chess.com...")
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        
        # Convertir en BGR pour OpenCV
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Recherche de motifs caractéristiques de Chess.com
        # (couleurs vertes/marron de l'échiquier)
        hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
        
        # Masque pour les cases vertes de Chess.com
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chercher le plus grand carré (l'échiquier)
        max_area = 0
        best_rect = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > 10000:  # Surface minimale
                x, y, w, h = cv2.boundingRect(contour)
                # Vérifier que c'est approximativement un carré
                if 0.8 < w/h < 1.2 and w > 300:
                    max_area = area
                    best_rect = (x, y, w, h)
        
        if best_rect:
            self.board_position = best_rect
            print(f"✓ Échiquier détecté à: x={best_rect[0]}, y={best_rect[1]}, taille={best_rect[2]}x{best_rect[3]}")
            return True
        else:
            print("✗ Échiquier non détecté. Assurez-vous que Chess.com est visible à l'écran.")
            return False
    
    def detect_board_state(self):
        """Détecte l'état actuel de l'échiquier (simplifié)"""
        # IMPORTANT: Cette version retourne une position de départ
        # Pour une vraie détection, il faudrait utiliser de la vision par ordinateur avancée
        print("⚠ Utilisation de la position de départ (détection complète non implémentée)")
        return chess.Board()
    
    def get_best_moves(self, board, num_moves=2):
        """Obtient les meilleurs coups depuis Stockfish"""
        if not self.engine:
            return []
        
        print("\n🤔 Analyse en cours...")
        result = self.engine.analyse(board, chess.engine.Limit(time=1.0), multipv=num_moves)
        
        moves = []
        for i, info in enumerate(result):
            move = info['pv'][0]
            score = info.get('score')
            moves.append({
                'move': move,
                'score': score,
                'rank': i + 1
            })
            print(f"  {i+1}. {move} (score: {score})")
        
        return moves
    
    def draw_arrow(self, image, from_square, to_square, color, board_rect):
        """Dessine une flèche sur l'échiquier"""
        x, y, w, h = board_rect
        square_size = w // 8
        
        # Convertir les coordonnées d'échecs en pixels
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        to_file = chess.square_file(to_square)
        to_rank = chess.square_rank(to_square)
        
        # Calculer les positions (depuis le bas pour les blancs)
        from_x = x + (from_file + 0.5) * square_size
        from_y = y + (7 - from_rank + 0.5) * square_size
        to_x = x + (to_file + 0.5) * square_size
        to_y = y + (7 - to_rank + 0.5) * square_size
        
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Dessiner la flèche
        arrow_width = square_size // 4
        draw.line([(from_x, from_y), (to_x, to_y)], fill=color, width=arrow_width)
        
        # Dessiner la pointe de la flèche
        angle = np.arctan2(to_y - from_y, to_x - from_x)
        arrow_length = square_size // 2
        
        # Points de la pointe
        p1 = (to_x, to_y)
        p2 = (to_x - arrow_length * np.cos(angle - np.pi/6),
              to_y - arrow_length * np.sin(angle - np.pi/6))
        p3 = (to_x - arrow_length * np.cos(angle + np.pi/6),
              to_y - arrow_length * np.sin(angle + np.pi/6))
        
        draw.polygon([p1, p2, p3], fill=color)
    
    def show_moves(self, moves):
        """Affiche les coups recommandés sur l'écran"""
        if not self.board_position or not moves:
            return
        
        screenshot = pyautogui.screenshot()
        
        colors = [
            (0, 100, 255, 180),  # Bleu pour le meilleur coup
            (255, 50, 50, 180),  # Rouge pour le second
        ]
        
        for i, move_info in enumerate(moves[:2]):
            move = move_info['move']
            color = colors[i]
            self.draw_arrow(screenshot, move.from_square, move.to_square, color, self.board_position)
        
        # Afficher l'image
        screenshot.show()
        print("\n✓ Coups affichés! Fermez l'image pour continuer.")
    
    def run(self):
        """Lance le détecteur"""
        print("=" * 60)
        print("🎯 CHESS.COM MOVE SUGGESTER")
        print("=" * 60)
        
        if not self.start_engine():
            return
        
        try:
            while True:
                input("\n📸 Appuyez sur ENTRÉE pour analyser l'échiquier (Ctrl+C pour quitter)...")
                
                if self.find_chessboard():
                    board = self.detect_board_state()
                    print(f"\n📋 Position actuelle:\n{board}")
                    
                    moves = self.get_best_moves(board)
                    if moves:
                        self.show_moves(moves)
                else:
                    print("Réessayez avec Chess.com visible à l'écran.")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme.")
        finally:
            if self.engine:
                self.engine.quit()
                print("✓ Moteur fermé.")

if __name__ == "__main__":
    # IMPORTANT: Remplacez ce chemin par celui de stockfish.exe
    stockfish_path = r""C:\Users\natha\botfish\stockfish\stockfish-windows-x86-64-avx2.exe""
    
    detector = ChessComDetector(stockfish_path)
    detector.run()
