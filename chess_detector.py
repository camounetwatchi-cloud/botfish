import chess
import chess.engine
import pyautogui
import numpy as np
from PIL import Image
import time
import cv2
import hashlib
import re

class ChessComDetector:
    def __init__(self, stockfish_path):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.board_position = None
        self.last_board_hash = None
        self.current_fen = None
        
    def start_engine(self):
        """Démarre le moteur Stockfish"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print("✓ Moteur Stockfish démarré")
            return True
        except Exception as e:
            print(f"✗ Erreur lors du démarrage de Stockfish: {e}")
            return False
    
    def find_chessboard(self):
        """Détecte la position de l'échiquier Chess.com à l'écran"""
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
        
        # Masque pour les cases vertes de Chess.com
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
    
    def extract_fen_from_page(self):
        """Cherche la notation FEN dans la page Chess.com"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
            
            # Chercher la section avec les coups (zone de texte)
            # Chess.com affiche parfois le FEN dans le DOM ou dans les outils développeur
            # Cette approche est limitée, on utilisera plutôt une entrée manuelle
            
            return None
        except:
            return None
    
    def manual_fen_input(self):
        """Permet à l'utilisateur d'entrer manuellement le FEN"""
        print("\n" + "="*60)
        print("⚠️  DÉTECTION AUTOMATIQUE NON DISPONIBLE")
        print("="*60)
        print("\nPour obtenir la position FEN sur Chess.com:")
        print("1. Faites un clic droit sur l'échiquier")
        print("2. Sélectionnez 'Copier FEN' ou 'Copy FEN'")
        print("3. Collez le FEN ci-dessous (ou tapez 'start' pour position initiale)")
        print("\n" + "="*60)
        
        fen_input = input("\n📋 Entrez le FEN: ").strip()
        
        if fen_input.lower() == 'start':
            return chess.STARTING_FEN
        
        # Valider le FEN
        try:
            board = chess.Board(fen_input)
            self.current_fen = fen_input
            return fen_input
        except:
            print("❌ FEN invalide, utilisation de la position de départ")
            return chess.STARTING_FEN
    
    def detect_board_state(self):
        """Détecte l'état actuel de l'échiquier"""
        # Essayer d'extraire le FEN automatiquement (non implémenté complètement)
        auto_fen = self.extract_fen_from_page()
        
        if auto_fen:
            return chess.Board(auto_fen)
        
        # Si pas de FEN stocké, demander à l'utilisateur
        if self.current_fen is None:
            fen = self.manual_fen_input()
            return chess.Board(fen)
        
        # Utiliser le FEN actuel
        return chess.Board(self.current_fen)
    
    def update_position_after_move(self, board, move):
        """Met à jour la position après un coup"""
        try:
            board.push(move)
            self.current_fen = board.fen()
            return board
        except:
            return board
    
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
    
    def print_moves(self, moves, board):
        """Affiche les meilleurs coups dans le terminal"""
        if not moves:
            return
        
        print("\n" + "="*60)
        print(f"♟️  POSITION ACTUELLE: {'Blancs' if board.turn else 'Noirs'} à jouer")
        print("="*60)
        print(board)
        print("\n" + "="*60)
        print("🎯 MEILLEURS COUPS:")
        print("="*60)
        
        for move_info in moves:
            move = move_info['move']
            score = move_info['score']
            rank = move_info['rank']
            
            if rank == 1:
                emoji = "🥇"
            elif rank == 2:
                emoji = "🥈"
            elif rank == 3:
                emoji = "🥉"
            else:
                emoji = f"{rank}."
            
            # Convertir le coup en notation lisible
            move_san = board.san(move)
            print(f"{emoji} {move_san} [{move}] (Score: {score})")
        
        print("="*60)
    
    def run(self):
        """Lance le détecteur en mode surveillance continue"""
        print("=" * 60)
        print("🎯 CHESS.COM MOVE SUGGESTER - MODE AUTO")
        print("=" * 60)
        print("\n⏳ Surveillance en continu activée...")
        print("💡 Le programme détecte automatiquement les changements visuels")
        print("🛑 Appuyez sur Ctrl+C pour arrêter")
        print("🔄 Tapez 'update' + Entrée pour changer la position manuellement\n")
        
        if not self.start_engine():
            return
        
        # Recherche initiale de l'échiquier
        print("🔍 Recherche de l'échiquier...")
        while not self.find_chessboard():
            print("⏳ Échiquier non détecté, nouvelle tentative dans 2s...")
            time.sleep(2)
        
        print("✓ Échiquier détecté!")
        print(f"📍 Position: x={self.board_position[0]}, y={self.board_position[1]}, taille={self.board_position[2]}x{self.board_position[3]}\n")
        
        # Obtenir la position initiale
        board = self.detect_board_state()
        
        # Première analyse
        print("⚡ Analyse initiale...")
        moves = self.get_best_moves(board)
        if moves:
            self.print_moves(moves, board)
        
        print("\n👀 Surveillance active... (détection visuelle des changements)")
        print("💡 Astuce: Après avoir joué, le changement sera détecté automatiquement")
        print("⚠️  Si la position n'est pas correcte, redémarrez et entrez le bon FEN\n")
        
        try:
            check_count = 0
            while True:
                check_count += 1
                
                # Vérifier si l'échiquier a changé visuellement
                if self.has_board_changed():
                    print(f"\n🔄 Changement visuel détecté! (vérification #{check_count})")
                    
                    # Demander à l'utilisateur de confirmer/entrer la nouvelle position
                    print("📝 Entrez le nouveau FEN (ou 'skip' pour ignorer, 'auto' pour tenter analyse auto):")
                    user_input = input(">>> ").strip()
                    
                    if user_input.lower() == 'skip':
                        continue
                    elif user_input.lower() == 'auto':
                        # Ici on pourrait ajouter une vraie détection OCR
                        print("⚠️  Fonction non disponible, utilisation de la position actuelle")
                        board = chess.Board(self.current_fen) if self.current_fen else chess.Board()
                    else:
                        try:
                            board = chess.Board(user_input)
                            self.current_fen = user_input
                            print("✓ Position mise à jour!")
                        except:
                            print("❌ FEN invalide, position inchangée")
                            continue
                    
                    print("⚡ Analyse en cours...")
                    moves = self.get_best_moves(board)
                    
                    if moves:
                        self.print_moves(moves, board)
                    
                    print("\n👀 Surveillance active...")
                
                # Attendre un peu avant la prochaine vérification
                time.sleep(1)
                
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
