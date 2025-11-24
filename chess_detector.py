import chess
import chess.engine
import pyautogui
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2
import hashlib

class ChessComDetector:
    def __init__(self, stockfish_path):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.board_position = None
        self.last_board_hash = None
        
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
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        
        # Convertir en BGR pour OpenCV
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Recherche de motifs caractéristiques de Chess.com
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
            if area > max_area and area > 10000:
                x, y, w, h = cv2.boundingRect(contour)
                if 0.8 < w/h < 1.2 and w > 300:
                    max_area = area
                    best_rect = (x, y, w, h)
        
        if best_rect:
            self.board_position = best_rect
            return True
        return False
    
    def get_board_hash(self):
        """Génère un hash de l'échiquier pour détecter les changements"""
        if not self.board_position:
            return None
        
        screenshot = pyautogui.screenshot()
        x, y, w, h = self.board_position
        board_img = screenshot.crop((x, y, x + w, y + h))
        
        # Convertir en array numpy et calculer un hash
        img_array = np.array(board_img)
        img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        return img_hash
    
    def has_board_changed(self):
        """Vérifie si l'échiquier a changé depuis la dernière analyse"""
        current_hash = self.get_board_hash()
        if current_hash is None:
            return False
        
        if self.last_board_hash is None:
            self.last_board_hash = current_hash
            return True
        
        if current_hash != self.last_board_hash:
            self.last_board_hash = current_hash
            return True
        
        return False
    
    def detect_board_state(self):
        """Détecte l'état actuel de l'échiquier (simplifié)"""
        # IMPORTANT: Cette version retourne une position de départ
        # Pour une vraie détection, il faudrait utiliser de la vision par ordinateur avancée
        return chess.Board()
    
    def get_best_moves(self, board, num_moves=3):
        """Obtient les meilleurs coups depuis Stockfish"""
        if not self.engine:
            return []
        
        result = self.engine.analyse(board, chess.engine.Limit(time=0.5), multipv=num_moves)
        
        moves = []
        for i, info in enumerate(result):
            move = info['pv'][0]
            score = info.get('score')
            moves.append({
                'move': move,
                'score': score,
                'rank': i + 1
            })
        
        return moves
    
    def print_moves(self, moves):
        """Affiche les meilleurs coups dans le terminal"""
        if not moves:
            return
        
        print("\n" + "="*50)
        print("🎯 MEILLEURS COUPS:")
        print("="*50)
        
        for move_info in moves:
            move = move_info['move']
            score = move_info['score']
            rank = move_info['rank']
            
            # Formatage de l'affichage
            if rank == 1:
                emoji = "🥇"
            elif rank == 2:
                emoji = "🥈"
            elif rank == 3:
                emoji = "🥉"
            else:
                emoji = f"{rank}."
            
            print(f"{emoji} {move} (Score: {score})")
        
        print("="*50)
    
    def run(self):
        """Lance le détecteur en mode surveillance continue"""
        print("=" * 60)
        print("🎯 CHESS.COM MOVE SUGGESTER - MODE AUTO")
        print("=" * 60)
        print("\n⏳ Surveillance en continu activée...")
        print("💡 Le programme détecte automatiquement les nouveaux coups")
        print("🛑 Appuyez sur Ctrl+C pour arrêter\n")
        
        if not self.start_engine():
            return
        
        # Recherche initiale de l'échiquier
        print("🔍 Recherche de l'échiquier...")
        while not self.find_chessboard():
            print("⏳ Échiquier non détecté, nouvelle tentative dans 2s...")
            time.sleep(2)
        
        print("✓ Échiquier détecté!")
        print(f"📍 Position: x={self.board_position[0]}, y={self.board_position[1]}, taille={self.board_position[2]}x{self.board_position[3]}\n")
        print("👀 Surveillance des changements...\n")
        
        try:
            check_count = 0
            while True:
                check_count += 1
                
                # Vérifier si l'échiquier a changé
                if self.has_board_changed():
                    print(f"\n🔄 Changement détecté! (vérification #{check_count})")
                    print("⚡ Analyse en cours...")
                    
                    board = self.detect_board_state()
                    moves = self.get_best_moves(board)
                    
                    if moves:
                        self.print_moves(moves)
                    
                    print("\n👀 Surveillance active...")
                
                # Attendre un peu avant la prochaine vérification
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme.")
        finally:
            if self.engine:
                self.engine.quit()
                print("✓ Moteur fermé.")

if __name__ == "__main__":
    # Chemin vers Stockfish
    stockfish_path = r"C:\Users\natha\botfish\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    detector = ChessComDetector(stockfish_path)
    detector.run()
