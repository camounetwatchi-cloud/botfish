import chess
import chess.engine
import pyautogui
import numpy as np
from PIL import Image
import time
import cv2
import hashlib
import os

class ChessComDetector:
    def __init__(self, stockfish_path):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.board_position = None
        self.last_board_hash = None
        self.last_fen = None
        self.piece_templates = {}
        self.square_size = 0
        
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
            self.square_size = best_rect[2] // 8
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
    
    def create_templates_from_board(self):
        """
        Créer des templates de pièces depuis la position de départ
        Cette fonction doit être appelée sur une position de départ connue
        """
        print("\n🎨 Création des templates de pièces...")
        print("⚠️  Assurez-vous que l'échiquier est en POSITION DE DÉPART")
        input("▶ Appuyez sur ENTRÉE quand prêt...")
        
        screenshot = pyautogui.screenshot()
        x, y, w, h = self.board_position
        
        # Définir où se trouvent les pièces en position de départ
        # Format: (file, rank): piece_symbol
        starting_pieces = {
            (0, 0): 'R', (1, 0): 'N', (2, 0): 'B', (3, 0): 'Q', 
            (4, 0): 'K', (5, 0): 'B', (6, 0): 'N', (7, 0): 'R',
            (0, 1): 'P', (1, 1): 'P', (2, 1): 'P', (3, 1): 'P',
            (4, 1): 'P', (5, 1): 'P', (6, 1): 'P', (7, 1): 'P',
            (0, 6): 'p', (1, 6): 'p', (2, 6): 'p', (3, 6): 'p',
            (4, 6): 'p', (5, 6): 'p', (6, 6): 'p', (7, 6): 'p',
            (0, 7): 'r', (1, 7): 'n', (2, 7): 'b', (3, 7): 'q',
            (4, 7): 'k', (5, 7): 'b', (6, 7): 'n', (7, 7): 'r',
        }
        
        for (file, rank), piece in starting_pieces.items():
            if piece not in self.piece_templates:
                # Extraire l'image de la case
                sx = x + file * self.square_size
                sy = y + (7 - rank) * self.square_size
                
                square_img = screenshot.crop((sx, sy, sx + self.square_size, sy + self.square_size))
                
                # Stocker le template
                self.piece_templates[piece] = np.array(square_img)
        
        # Extraire aussi une case vide (par exemple e4)
        sx = x + 4 * self.square_size
        sy = y + 4 * self.square_size
        empty_square = screenshot.crop((sx, sy, sx + self.square_size, sy + self.square_size))
        self.piece_templates['empty'] = np.array(empty_square)
        
        print(f"✓ {len(self.piece_templates)} templates créés!")
        
        # Sauvegarder les templates
        if not os.path.exists('templates'):
            os.makedirs('templates')
        
        for piece, template in self.piece_templates.items():
            img = Image.fromarray(template)
            img.save(f'templates/{piece}.png')
        
        print("✓ Templates sauvegardés dans le dossier 'templates/'")
    
    def load_templates(self):
        """Charge les templates depuis le dossier"""
        if not os.path.exists('templates'):
            return False
        
        template_files = os.listdir('templates')
        if len(template_files) == 0:
            return False
        
        for filename in template_files:
            if filename.endswith('.png'):
                piece = filename.replace('.png', '')
                img = Image.open(f'templates/{filename}')
                self.piece_templates[piece] = np.array(img)
        
        print(f"✓ {len(self.piece_templates)} templates chargés")
        return True
    
    def match_piece(self, square_img):
        """
        Compare une case avec tous les templates et retourne la meilleure correspondance
        """
        square_img_np = np.array(square_img)
        
        # Redimensionner si nécessaire
        if square_img_np.shape[:2] != (self.square_size, self.square_size):
            square_img_np = cv2.resize(square_img_np, (self.square_size, self.square_size))
        
        # Convertir en niveaux de gris
        square_gray = cv2.cvtColor(square_img_np, cv2.COLOR_RGB2GRAY)
        
        # D'abord, vérifier si la case est vide
        if 'empty' in self.piece_templates:
            empty_template_gray = cv2.cvtColor(self.piece_templates['empty'], cv2.COLOR_RGB2GRAY)
            empty_result = cv2.matchTemplate(square_gray, empty_template_gray, cv2.TM_CCOEFF_NORMED)
            empty_score = empty_result[0][0]
            
            # Si très similaire à une case vide, c'est vide
            if empty_score > 0.85:
                return None
        
        # Sinon, chercher quelle pièce c'est
        best_match = None
        best_score = 0
        
        for piece, template in self.piece_templates.items():
            if piece == 'empty':
                continue  # Déjà vérifié
            
            template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            
            # Calculer la similarité (corrélation)
            result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            
            if score > best_score:
                best_score = score
                best_match = piece
        
        # Seuil de confiance pour les pièces
        if best_score < 0.65:
            return None  # Probablement vide ou non reconnu
        
        return best_match
    
    def detect_board_state(self):
        """Détecte l'état complet de l'échiquier par reconnaissance de patterns"""
        if not self.board_position:
            return chess.Board()
        
        if not self.piece_templates:
            print("⚠️  Aucun template disponible!")
            return chess.Board()
        
        print("🔍 Analyse de l'échiquier en cours...")
        
        screenshot = pyautogui.screenshot()
        x, y, w, h = self.board_position
        
        # Créer une matrice 8x8 pour stocker les pièces
        board_matrix = [[None for _ in range(8)] for _ in range(8)]
        
        # Scanner toutes les cases
        pieces_found = 0
        for rank in range(8):
            for file in range(8):
                sx = x + file * self.square_size
                sy = y + (7 - rank) * self.square_size
                
                square_img = screenshot.crop((sx, sy, sx + self.square_size, sy + self.square_size))
                
                # Reconnaître la pièce
                piece = self.match_piece(square_img)
                board_matrix[rank][file] = piece
                
                if piece is not None:
                    pieces_found += 1
        
        print(f"   {pieces_found} pièces détectées")
        
        # Convertir la matrice en FEN
        fen = self.matrix_to_fen(board_matrix)
        
        print(f"📋 FEN détecté: {fen}")
        
        try:
            board = chess.Board(fen)
            self.last_fen = fen
            return board
        except Exception as e:
            print(f"⚠️  Erreur de détection FEN: {e}")
            print(f"    FEN généré: {fen}")
            
            # Afficher la matrice pour debug
            print("\n    Debug - Matrice détectée:")
            for rank in range(7, -1, -1):
                row = ""
                for file in range(8):
                    p = board_matrix[rank][file]
                    row += (p if p else '.') + " "
                print(f"    {rank+1}: {row}")
            
            if self.last_fen:
                print("\n    Utilisation de la dernière position connue")
                return chess.Board(self.last_fen)
            return chess.Board()
    
    
    def matrix_to_fen(self, matrix):
        """Convertit une matrice 8x8 de pièces en notation FEN"""
        fen_rows = []
        
        # Parcourir du rang 8 au rang 1 (inversé pour FEN)
        for rank in range(7, -1, -1):
            fen_row = ""
            empty_count = 0
            
            for file in range(8):
                piece = matrix[rank][file]
                
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        # Ajouter les métadonnées FEN
        fen = "/".join(fen_rows) + " w KQkq - 0 1"
        return fen
    
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
        print("🎯 CHESS.COM MOVE SUGGESTER - AUTO DETECTION")
        print("=" * 60)
        print("\n🤖 Reconnaissance automatique des pièces")
        print("🛑 Appuyez sur Ctrl+C pour arrêter\n")
        
        if not self.start_engine():
            return
        
        # Recherche initiale de l'échiquier
        print("🔍 Recherche de l'échiquier...")
        while not self.find_chessboard():
            print("⏳ Échiquier non détecté, nouvelle tentative dans 2s...")
            time.sleep(2)
        
        print("✓ Échiquier détecté!")
        print(f"📍 Position: x={self.board_position[0]}, y={self.board_position[1]}, taille={self.board_position[2]}x{self.board_position[3]}")
        print(f"📏 Taille case: {self.square_size}px\n")
        
        # Charger ou créer les templates
        if not self.load_templates():
            print("⚠️  Aucun template trouvé. Création nécessaire...")
            self.create_templates_from_board()
        
        # Première analyse
        print("\n⚡ Analyse initiale...")
        board = self.detect_board_state()
        moves = self.get_best_moves(board)
        if moves:
            self.print_moves(moves, board)
        
        print("\n👀 Surveillance active (vérification toutes les 2 secondes)...")
        
        try:
            check_count = 0
            while True:
                time.sleep(2)
                check_count += 1
                
                # Vérifier si l'échiquier a changé visuellement
                if self.has_board_changed():
                    print(f"\n🔄 Changement détecté! (#{check_count})")
                    
                    board = self.detect_board_state()
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
