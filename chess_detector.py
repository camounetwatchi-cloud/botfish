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
        
        # Mapping entre votre format de noms et les symboles FEN
        self.template_to_fen = {
            'wk': 'K', 'wq': 'Q', 'wr': 'R', 'wb': 'B', 'wn': 'N', 'wp': 'P',
            'bk': 'k', 'bq': 'q', 'br': 'r', 'bb': 'b', 'bn': 'n', 'bp': 'p'
        }
        
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
    
    def load_templates(self):
        """Charge les templates depuis le dossier avec votre format de noms"""
        if not os.path.exists('templates'):
            print("❌ Dossier 'templates/' introuvable!")
            return False
        
        template_files = os.listdir('templates')
        if len(template_files) == 0:
            print("❌ Aucun fichier dans 'templates/'!")
            return False
        
        # Liste des pièces attendues
        expected_templates = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp', 
                            'bk', 'bq', 'br', 'bb', 'bn', 'bp', 'empty']
        
        loaded_count = 0
        for template_name in expected_templates:
            filename = f'templates/{template_name}.png'
            if os.path.exists(filename):
                img = Image.open(filename)
                self.piece_templates[template_name] = np.array(img)
                loaded_count += 1
            else:
                print(f"⚠️  Template manquant: {filename}")
        
        if loaded_count < 13:
            print(f"❌ Templates incomplets: {loaded_count}/13 trouvés")
            print(f"   Fichiers attendus dans 'templates/': wk.png, wq.png, wr.png, wb.png, wn.png, wp.png,")
            print(f"                                          bk.png, bq.png, br.png, bb.png, bn.png, bp.png, empty.png")
            return False
        
        print(f"✅ {loaded_count} templates chargés depuis 'templates/'")
        print(f"   Pièces blanches: wk, wq, wr, wb, wn, wp")
        print(f"   Pièces noires: bk, bq, br, bb, bn, bp")
        print(f"   Case vide: empty")
        return True
    
    def match_piece(self, square_img):
        """Compare une case avec tous les templates et retourne la meilleure correspondance"""
        square_img_np = np.array(square_img)
        
        # Redimensionner le template à la taille de la case détectée
        target_size = square_img_np.shape[:2]
        
        # Convertir en niveaux de gris
        square_gray = cv2.cvtColor(square_img_np, cv2.COLOR_RGB2GRAY)
        
        # D'abord, vérifier si la case est vide
        if 'empty' in self.piece_templates:
            empty_template = self.piece_templates['empty']
            empty_resized = cv2.resize(empty_template, (target_size[1], target_size[0]))
            empty_template_gray = cv2.cvtColor(empty_resized, cv2.COLOR_RGB2GRAY)
            
            empty_result = cv2.matchTemplate(square_gray, empty_template_gray, cv2.TM_CCOEFF_NORMED)
            empty_score = empty_result[0][0]
            
            # Si très similaire à une case vide
            if empty_score > 0.85:
                return None
        
        # Chercher quelle pièce correspond le mieux
        best_match = None
        best_score = 0
        
        for template_name, template in self.piece_templates.items():
            if template_name == 'empty':
                continue
            
            # Redimensionner le template à la taille de la case
            template_resized = cv2.resize(template, (target_size[1], target_size[0]))
            template_gray = cv2.cvtColor(template_resized, cv2.COLOR_RGB2GRAY)
            
            # Calculer la similarité
            result = cv2.matchTemplate(square_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            
            if score > best_score:
                best_score = score
                best_match = template_name
        
        # Seuil de confiance
        if best_score < 0.60:
            return None
        
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
        pieces_found = {'white': 0, 'black': 0}
        detected_pieces = []
        
        for rank in range(8):
            for file in range(8):
                sx = x + file * self.square_size
                sy = y + (7 - rank) * self.square_size
                
                square_img = screenshot.crop((sx, sy, sx + self.square_size, sy + self.square_size))
                
                # Reconnaître la pièce
                piece_template = self.match_piece(square_img)
                
                if piece_template:
                    # Convertir le format de template (wk, bp, etc.) en symbole FEN (K, p, etc.)
                    fen_symbol = self.template_to_fen[piece_template]
                    board_matrix[rank][file] = fen_symbol
                    detected_pieces.append(f"{piece_template}@{chr(97+file)}{rank+1}")
                    
                    if piece_template[0] == 'w':
                        pieces_found['white'] += 1
                    else:
                        pieces_found['black'] += 1
        
        total_pieces = pieces_found['white'] + pieces_found['black']
        print(f"   {total_pieces} pièces détectées (Blanches: {pieces_found['white']}, Noires: {pieces_found['black']})")
        
        # Afficher les pièces détectées pour debug
        if total_pieces < 20:
            print(f"   ⚠️ Peu de pièces détectées. Pièces trouvées: {', '.join(detected_pieces[:10])}{'...' if len(detected_pieces) > 10 else ''}")
        
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
            print("      a b c d e f g h")
            for rank in range(7, -1, -1):
                row = f"    {rank+1} "
                for file in range(8):
                    p = board_matrix[rank][file]
                    row += (p if p else '.') + " "
                print(row)
            
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
        print("\n🤖 Reconnaissance automatique avec templates fixes")
        print("🛑 Appuyez sur Ctrl+C pour arrêter\n")
        
        if not self.start_engine():
            return
        
        # Charger les templates
        if not self.load_templates():
            print("\n❌ Impossible de charger les templates!")
            print("Assurez-vous d'avoir un dossier 'templates/' avec les fichiers:")
            print("   wk.png, wq.png, wr.png, wb.png, wn.png, wp.png,")
            print("   bk.png, bq.png, br.png, bb.png, bn.png, bp.png, empty.png")
            return
        
        # Recherche initiale de l'échiquier
        print("\n🔍 Recherche de l'échiquier...")
        while not self.find_chessboard():
            print("⏳ Échiquier non détecté, nouvelle tentative dans 2s...")
            time.sleep(2)
        
        print("✓ Échiquier détecté!")
        print(f"📍 Position: x={self.board_position[0]}, y={self.board_position[1]}, taille={self.board_position[2]}x{self.board_position[3]}")
        print(f"📏 Taille case: {self.square_size}px\n")
        
        # Première analyse
        print("⚡ Analyse initiale...")
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
    # Chemin vers Stockfish - MODIFIEZ CE CHEMIN
    stockfish_path = r"C:\Users\natha\botfish\stockfish\stockfish-windows-x86-64-avx2.exe"
    
    detector = ChessComDetector(stockfish_path)
    detector.run()
