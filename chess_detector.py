import chess
import chess.engine
import pyautogui
import numpy as np
from PIL import Image
import time
import cv2
import hashlib

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
    
    def get_square_image(self, screenshot, square_index):
        """Extrait l'image d'une case spécifique (0-63)"""
        if not self.board_position:
            return None
        
        x, y, w, h = self.board_position
        square_size = w // 8
        
        # Calculer la position de la case (a1 = 0, h8 = 63)
        file = square_index % 8  # colonne (0-7)
        rank = square_index // 8  # rangée (0-7)
        
        # Coordonnées en pixels (du point de vue des blancs en bas)
        sx = x + file * square_size
        sy = y + (7 - rank) * square_size
        
        square_img = screenshot.crop((sx, sy, sx + square_size, sy + square_size))
        return np.array(square_img)
    
    def detect_piece_on_square(self, square_img):
        """
        Détecte quelle pièce est sur une case
        Retourne: 'P','N','B','R','Q','K' (blanc) ou 'p','n','b','r','q','k' (noir) ou None
        """
        if square_img is None:
            return None
        
        # Convertir en HSV
        hsv = cv2.cvtColor(square_img, cv2.COLOR_RGB2HSV)
        
        # Calculer la luminosité moyenne de la case
        brightness = np.mean(hsv[:, :, 2])
        
        # Détecter si une pièce est présente (zone sombre au centre)
        center_h = square_img.shape[0] // 2
        center_w = square_img.shape[1] // 2
        margin = square_img.shape[0] // 4
        
        center_region = square_img[
            center_h - margin:center_h + margin,
            center_w - margin:center_w + margin
        ]
        
        # Calculer la densité de pixels "pièce" (ni trop clair, ni la couleur de la case)
        gray_center = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY)
        
        # Seuillage pour détecter la présence d'une pièce
        _, thresh = cv2.threshold(gray_center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        piece_pixels = np.sum(thresh < 128)
        total_pixels = thresh.size
        piece_ratio = piece_pixels / total_pixels
        
        # Si moins de 15% de pixels "pièce", la case est vide
        if piece_ratio < 0.15:
            return None
        
        # Détecter la couleur de la pièce
        # Les pièces blanches ont plus de luminosité
        piece_region_hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        avg_brightness = np.mean(piece_region_hsv[:, :, 2])
        
        is_white = avg_brightness > 130  # Seuil empirique
        
        # Pour le type de pièce, on utilise la forme (simplification)
        # Ici on utilise une heuristique basique:
        # - Hauteur de la pièce (roi/reine = grand, pion = petit)
        # - Largeur (cavalier/fou différents)
        
        # Trouver les contours de la pièce
        edges = cv2.Canny(gray_center, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Si on détecte une pièce mais pas de contours, supposer un pion
            return 'P' if is_white else 'p'
        
        # Prendre le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Heuristiques simples (à améliorer avec ML)
        aspect_ratio = h / w if w > 0 else 1
        area = cv2.contourArea(largest_contour)
        
        # Classification simplifiée (très approximative)
        if aspect_ratio > 2.5:  # Très vertical
            piece_type = 'K' if is_white else 'k'  # Roi ou Reine
        elif aspect_ratio > 2.0:
            piece_type = 'Q' if is_white else 'q'
        elif aspect_ratio > 1.5:
            piece_type = 'R' if is_white else 'r'  # Tour
        elif aspect_ratio > 1.2:
            piece_type = 'B' if is_white else 'b'  # Fou
        elif area < 500:  # Petit
            piece_type = 'P' if is_white else 'p'  # Pion
        else:
            piece_type = 'N' if is_white else 'n'  # Cavalier
        
        return piece_type
    
    def detect_board_state(self):
        """Détecte l'état complet de l'échiquier"""
        if not self.board_position:
            return chess.Board()
        
        print("🔍 Analyse de l'échiquier en cours...")
        
        screenshot = pyautogui.screenshot()
        
        # Créer une matrice 8x8 pour stocker les pièces
        board_matrix = [[None for _ in range(8)] for _ in range(8)]
        
        # Scanner toutes les cases
        for square_index in range(64):
            square_img = self.get_square_image(screenshot, square_index)
            piece = self.detect_piece_on_square(square_img)
            
            rank = square_index // 8
            file = square_index % 8
            board_matrix[rank][file] = piece
        
        # Convertir la matrice en FEN
        fen = self.matrix_to_fen(board_matrix)
        
        print(f"📋 FEN détecté: {fen}")
        
        try:
            board = chess.Board(fen)
            self.last_fen = fen
            return board
        except Exception as e:
            print(f"⚠️  Erreur de détection FEN: {e}")
            print("   Utilisation de la dernière position connue ou position initiale")
            if self.last_fen:
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
        
        # Ajouter les métadonnées FEN (simplifié: blancs au trait, tous les roques possibles)
        fen = "/".join(fen_rows) + " w KQkq - 0 1"
        return fen
    
    def get_best_moves(self, board, num_moves=3):
        """Obtient les meilleurs coups depuis Stockfish"""
        if not self.engine:
            return []
        
        try:
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
        except Exception as e:
            print(f"⚠️  Erreur d'analyse: {e}")
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
            
            move_san = board.san(move)
            print(f"{emoji} {move_san} [{move}] (Score: {score})")
        
        print("="*60)
    
    def run(self):
        """Lance le détecteur en mode surveillance continue"""
        print("=" * 60)
        print("🎯 CHESS.COM MOVE SUGGESTER - DÉTECTION AUTO")
        print("=" * 60)
        print("\n🤖 Détection automatique des pièces activée")
        print("⏳ Surveillance en continu...")
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
        
        # Première analyse
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
                self.engine.quit()
                print("✓ Moteur fermé.")

if __name__ == "__main__":
    # Chemin vers Stockfish
    stockfish_path = r"C:\Users\natha\botfish\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    detector = ChessComDetector(stockfish_path)
    detector.run()
