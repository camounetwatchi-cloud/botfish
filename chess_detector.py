import chess
import chess.engine
import pyautogui
import numpy as np
import time
import cv2
import hashlib
import re
import pyperclip
import keyboard

class ChessComDetector:
    def __init__(self, stockfish_path):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.board_position = None
        self.last_board_hash = None
        self.last_fen = None
        
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
    
    def extract_fen_from_clipboard(self):
        """
        Essaie d'extraire le FEN depuis le presse-papiers
        L'utilisateur doit faire clic droit > Copier FEN sur Chess.com
        """
        try:
            clipboard_content = pyperclip.paste()
            
            # Vérifier si c'est un FEN valide
            if clipboard_content and len(clipboard_content) > 10:
                # Pattern FEN basique
                if re.match(r'^[rnbqkpRNBQKP1-8/\s]+\s[wb]\s', clipboard_content):
                    return clipboard_content.strip()
        except Exception as e:
            print(f"⚠️  Erreur clipboard: {e}")
        
        return None
    
    def auto_copy_fen(self):
        """
        Automatise la copie du FEN depuis Chess.com
        Simule un clic droit sur l'échiquier et sélectionne "Copier FEN"
        """
        if not self.board_position:
            return None
        
        try:
            x, y, w, h = self.board_position
            center_x = x + w // 2
            center_y = y + h // 2
            
            print("🖱️  Clic droit sur l'échiquier...")
            
            # Faire un clic droit au centre de l'échiquier
            pyautogui.rightClick(center_x, center_y)
            time.sleep(0.3)
            
            # Chess.com affiche "Copy FEN" dans le menu
            # On simule la touche 'c' ou on cherche le texte
            # Essai 1: Appuyer sur 'c' (si c'est le raccourci)
            pyautogui.press('c')
            time.sleep(0.2)
            
            # Vérifier si le FEN est dans le clipboard
            fen = self.extract_fen_from_clipboard()
            
            if fen:
                print(f"✓ FEN copié automatiquement!")
                return fen
            
            # Essai 2: Cliquer sur la position du menu
            # (Position approximative, peut varier)
            print("   Tentative de clic sur 'Copy FEN'...")
            pyautogui.click(center_x, center_y + 60)
            time.sleep(0.2)
            
            fen = self.extract_fen_from_clipboard()
            if fen:
                print(f"✓ FEN copié!")
                return fen
            
            print("⚠️  Copie automatique échouée")
            return None
            
        except Exception as e:
            print(f"⚠️  Erreur auto-copy: {e}")
            return None
    
    def prompt_for_fen(self):
        """Demande à l'utilisateur de copier manuellement le FEN"""
        print("\n" + "="*60)
        print("📋 COPIE DU FEN")
        print("="*60)
        print("\n1. Faites un CLIC DROIT sur l'échiquier Chess.com")
        print("2. Cliquez sur 'Copy FEN' / 'Copier FEN'")
        print("3. Appuyez sur ENTRÉE ici")
        print("\nOu tapez 'start' pour la position initiale")
        print("="*60)
        
        input("\n▶ Appuyez sur ENTRÉE après avoir copié le FEN...")
        
        # Lire depuis le clipboard
        fen = self.extract_fen_from_clipboard()
        
        if fen:
            print(f"✓ FEN détecté: {fen[:50]}...")
            return fen
        
        # Sinon demander une saisie manuelle
        print("\n⚠️  Aucun FEN détecté dans le presse-papiers")
        fen_input = input("📝 Collez ou tapez le FEN: ").strip()
        
        if fen_input.lower() == 'start':
            return chess.STARTING_FEN
        
        return fen_input if fen_input else chess.STARTING_FEN
    
    def detect_board_state(self, auto_copy=False):
        """Détecte l'état actuel de l'échiquier"""
        
        if auto_copy:
            # Essayer la copie automatique
            fen = self.auto_copy_fen()
            if fen:
                self.last_fen = fen
                return chess.Board(fen)
        
        # Sinon, demander à l'utilisateur
        fen = self.prompt_for_fen()
        
        try:
            board = chess.Board(fen)
            self.last_fen = fen
            return board
        except Exception as e:
            print(f"❌ FEN invalide: {e}")
            print("   Utilisation de la position de départ")
            return chess.Board()
    
    def get_best_moves(self, board, num_moves=3):
        """Obtient les meilleurs coups depuis Stockfish"""
        if not self.engine:
            return []
        
        try:
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
            
            return moves
        except Exception as e:
            print(f"⚠️  Erreur d'analyse: {e}")
            # Redémarrer le moteur si nécessaire
            try:
                self.engine.quit()
            except:
                pass
            self.start_engine()
            return []
    
    def print_moves(self, moves, board):
        """Affiche les meilleurs coups dans le terminal"""
        if not moves:
            return
        
        print("\n" + "="*60)
        print(f"♟️  POSITION: {'Blancs' if board.turn else 'Noirs'} à jouer")
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
            
            try:
                move_san = board.san(move)
                print(f"{emoji} {move_san} [{move}] (Score: {score})")
            except:
                print(f"{emoji} {move} (Score: {score})")
        
        print("="*60)
    
    def run(self):
        """Lance le détecteur en mode surveillance continue"""
        print("=" * 60)
        print("🎯 CHESS.COM MOVE SUGGESTER")
        print("=" * 60)
        print("\n⏳ Mode surveillance avec copie FEN")
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
        
        # Obtenir la position initiale
        print("📋 Obtention de la position initiale...")
        board = self.detect_board_state(auto_copy=False)
        
        # Première analyse
        print("\n⚡ Analyse initiale...")
        moves = self.get_best_moves(board)
        if moves:
            self.print_moves(moves, board)
        
        print("\n👀 Surveillance active...")
        print("💡 Quand un coup est joué, le changement sera détecté automatiquement")
        
        try:
            check_count = 0
            while True:
                time.sleep(1.5)
                check_count += 1
                
                # Vérifier si l'échiquier a changé visuellement
                if self.has_board_changed():
                    print(f"\n🔄 Changement détecté! (#{check_count})")
                    
                    # Essayer copie automatique
                    print("📋 Récupération de la nouvelle position...")
                    board = self.detect_board_state(auto_copy=True)
                    
                    print("⚡ Analyse en cours...")
                    moves = self.get_best_moves(board)
                    
                    if moves:
                        self.print_moves(moves, board)
                    
                    print("\n👀 Surveillance active...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme.")
        finally:
            if self.engine:
                try:
                    self.engine.quit()
                    print("✓ Moteur fermé.")
                except:
                    pass

if __name__ == "__main__":
    # Chemin vers Stockfish
    stockfish_path = r"C:\Users\natha\botfish\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    detector = ChessComDetector(stockfish_path)
    detector.run()
