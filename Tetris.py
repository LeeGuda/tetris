import pygame
import random
import numpy as np

# --- 1. 상수 정의 (TetrisEnv에서 임포트함) ---
SQUARE_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SIDEBAR_GRID_WIDTH = 6
SCREEN_WIDTH = BOARD_WIDTH * SQUARE_SIZE
SIDEBAR_WIDTH = SIDEBAR_GRID_WIDTH * SQUARE_SIZE
FULL_SCREEN_WIDTH = SCREEN_WIDTH + SIDEBAR_WIDTH

# RGB 색상
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

COLORS = [
    (0, 255, 255),  # 0: Cyan (I)
    (0, 0, 255),    # 1: Blue (J)
    (255, 165, 0),  # 2: Orange (L)
    (255, 255, 0),  # 3: Yellow (O)
    (0, 255, 0),    # 4: Green (S)
    (128, 0, 128),  # 5: Purple (T)
    (255, 0, 0)     # 6: Red (Z)
]

# --- 2. 테트로미노 모양 정의 (TetrisEnv에서 임포트함) ---
TETROMINOS = [
    # 0: I (값 1)
    [[[1, 1, 1, 1]], [[1], [1], [1], [1]]], 
    # 1: J (값 2)
    [[[2, 0, 0], [2, 2, 2]], [[2, 2], [2, 0], [2, 0]], [[2, 2, 2], [0, 0, 2]], [[0, 2], [0, 2], [2, 2]]], 
    # 2: L (값 3)
    [[[0, 0, 3], [3, 3, 3]], [[3, 0], [3, 0], [3, 3]], [[3, 3, 3], [3, 0, 0]], [[3, 3], [0, 3], [0, 3]]],
    # 3: O (값 4)
    [[[4, 4], [4, 4]]], 
    # 4: S (값 5)
    [[[0, 5, 5], [5, 5, 0]], [[5, 0], [5, 5], [0, 5]]],
    # 5: T (값 6)
    [[[0, 6, 0], [6, 6, 6]], [[6, 0], [6, 6], [6, 0]], [[6, 6, 6], [0, 6, 0]], [[0, 6], [6, 6], [0, 6]]],
    # 6: Z (값 7 -> 6으로 수정) 👈 🚨 버그 수정
    [[[6, 6, 0], [0, 6, 6]], [[0, 6], [6, 6], [6, 0]]] 
]

# --- 3. 핵심 유틸리티 함수 (TetrisEnv에서 임포트함) ---

def generate_random_block_index():
    """다음 블록을 무작위로 생성합니다."""
    return random.randint(0, len(TETROMINOS) - 1)

def check_collision(board, shape, x, y):
    """주어진 위치와 모양으로 보드에 충돌하는지 확인합니다."""
    for row_idx, row in enumerate(shape):
        for col_idx, cell in enumerate(row):
            if cell != 0:
                board_x = x + col_idx
                board_y = y + row_idx
                
                # 1. 보드 경계 확인 (좌우, 바닥)
                if board_x < 0 or board_x >= BOARD_WIDTH or board_y >= BOARD_HEIGHT:
                    return True
                
                # 2. 고정된 블록과의 충돌 확인 (Y < 0일 때 강제로 진행되는 로직 제거)
                # 💡 블록이 보드 영역(Y >= 0) 내에 있다면 충돌 검사를 수행합니다.
                if board_y >= 0: 
                    if board[board_y][board_x] != 0:
                        return True
                        
    return False

def get_ghost_y(board, block):
    """현재 블록이 최종적으로 착지할 Y 좌표를 계산합니다."""
    dy = 0
    while not check_collision(board, block.shape, block.x, block.y + dy + 1):
        dy += 1
    return block.y + dy

def clear_lines(board):
    """꽉 찬 줄을 제거하고 보드를 업데이트하며, 제거된 줄 수를 반환합니다."""
    lines_cleared = 0
    new_board = []
    
    for row in board:
        if 0 in row: 
            new_board.append(row)
        else:
            lines_cleared += 1
            
    for _ in range(lines_cleared):
        new_board.insert(0, [0] * BOARD_WIDTH)
        
    return lines_cleared, new_board

def lock_block(board, block): 
    """블록을 보드에 고정합니다."""
    for row_idx, row in enumerate(block.shape):
        for col_idx, cell in enumerate(row):
            if cell != 0:
                # 색상 인덱스(0-6) + 1을 하여 보드에 저장 (0은 빈 공간)
                board[block.y + row_idx][block.x + col_idx] = block.shape_index + 1


# --- 4. CurrentBlock 클래스 (TetrisEnv에서 임포트함) ---

class CurrentBlock:
    def __init__(self, board, shape_index): 
        self.shape_index = shape_index
        self.color_index = self.shape_index 
        self.rotation = 0 
        self.board = board 
        
        self.shape = TETROMINOS[self.shape_index][self.rotation]
        
        # 시작 X 좌표 설정 (기존 로직 유지)
        block_width = len(self.shape[0])
        self.x = (BOARD_WIDTH // 2) - (block_width // 2)
        
        # 💡 수정: 시작 Y 좌표를 보드 상단 바깥 (Y=-2 또는 -3)으로 설정
        # 이렇게 하면 충돌 로직이 안정화될 여지가 생깁니다.
        self.y = -2 
        
        # ⚠️ 경고: Y=-2는 블록이 상단 2줄 위에서 시작함을 의미합니다.

    def rotate_to(self, new_rotation):
        """AI의 행동(action)에 따라 블록의 회전 상태를 즉시 설정합니다."""
        max_rotations = len(TETROMINOS[self.shape_index])
        
        self.rotation = new_rotation % max_rotations
            
        self.shape = TETROMINOS[self.shape_index][self.rotation]

    def can_move(self, board, dx, dy):
        """주어진 방향(dx, dy)으로 이동이 가능한지 충돌을 확인합니다."""
        return not check_collision(board, self.shape, self.x + dx, self.y + dy)
        
    def move(self, dx, dy):
        """블록을 이동시킵니다. (GUI 모드에서 사용)"""
        if self.can_move(self.board, dx, dy):
            self.x += dx
            self.y += dy
            return True 
        return False 

# --- 5. 렌더링 도우미 함수 (TetrisEnv에서 임포트함) ---

def draw_block(surface, color, x, y, outline_only=False):
    """단일 미니 블록을 그립니다."""
    rect = (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
    if outline_only:
        pygame.draw.rect(surface, color, rect, 1)
    else:
        pygame.draw.rect(surface, color, rect, 0)
        pygame.draw.rect(surface, BLACK, rect, 1)

def draw_board(surface, board, current_block):
    """메인 보드와 현재 움직이는 블록, 고스트 블록을 그립니다."""
    
    # 1. 고스트 블록 그리기
    if current_block:
        ghost_y = get_ghost_y(board, current_block)
        ghost_color = COLORS[current_block.color_index]
        for y_offset, row in enumerate(current_block.shape):
            for x_offset, cell in enumerate(row):
                if cell != 0:
                    draw_block(surface, ghost_color, 
                               current_block.x + x_offset, 
                               ghost_y + y_offset, 
                               outline_only=True)
                               
    # 2. 고정된 블록 그리기
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x] != 0:
                color_index = board[y][x] - 1 
                draw_block(surface, COLORS[color_index], x, y)
                
    # 3. 현재 움직이는 블록 그리기
    if current_block:
        current_color = COLORS[current_block.color_index]
        for y_offset, row in enumerate(current_block.shape):
            for x_offset, cell in enumerate(row):
                if cell != 0:
                    draw_block(surface, current_color, 
                               current_block.x + x_offset, 
                               current_block.y + y_offset)

def draw_held_block(surface, held_index):
    """홀드 블록을 사이드바에 그립니다."""
    PREVIEW_START_X = BOARD_WIDTH + 1
    PREVIEW_START_Y = 1 
    
    font = pygame.font.Font(None, 30)
    text = font.render("HOLD", True, WHITE)
    title_x = SCREEN_WIDTH + (SIDEBAR_WIDTH // 2) - (text.get_width() // 2)
    surface.blit(text, (title_x, PREVIEW_START_Y * SQUARE_SIZE))

    if held_index != -1 and held_index is not None:
        shape_data = TETROMINOS[held_index][0]
        color_index = held_index
        
        block_width = len(shape_data[0])
        preview_area_width = SIDEBAR_GRID_WIDTH - 2
        center_x_offset = (preview_area_width - block_width) // 2
        draw_y_offset = PREVIEW_START_Y + 2
        
        for y_offset, row in enumerate(shape_data):
            for x_offset, cell in enumerate(row):
                if cell != 0:
                    draw_block(surface, COLORS[color_index], 
                               PREVIEW_START_X + center_x_offset + x_offset, 
                               draw_y_offset + y_offset)


def draw_next_blocks(surface, next_blocks_queue):
    """다음 블록 큐를 사이드바에 그립니다."""
    PREVIEW_START_X = BOARD_WIDTH + 1
    PREVIEW_START_Y = 7 
    
    font = pygame.font.Font(None, 30)
    text = font.render("NEXT", True, WHITE)
    title_x = SCREEN_WIDTH + (SIDEBAR_WIDTH // 2) - (text.get_width() // 2) 
    surface.blit(text, (title_x, PREVIEW_START_Y * SQUARE_SIZE))

    current_y_offset = PREVIEW_START_Y + 2

    for block_index in next_blocks_queue:
        shape_data = TETROMINOS[block_index][0]
        color_index = block_index
        
        block_width = len(shape_data[0])
        preview_area_width = SIDEBAR_GRID_WIDTH - 2
        center_x_offset = (preview_area_width - block_width) // 2
        
        for y_offset, row in enumerate(shape_data):
            for x_offset, cell in enumerate(row):
                if cell != 0:
                    draw_block(surface, COLORS[color_index], 
                               PREVIEW_START_X + center_x_offset + x_offset, 
                               current_y_offset + y_offset)
                                
            current_y_offset += len(shape_data) + 1