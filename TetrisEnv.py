from Tetris import (
    TETROMINOS, 
    CurrentBlock,
    draw_board,
    draw_held_block,
    draw_next_blocks, 
    generate_random_block_index, 
    get_ghost_y, 
    lock_block, 
    clear_lines, 
    COLORS, BLACK, WHITE, 
    SQUARE_SIZE, BOARD_WIDTH, BOARD_HEIGHT, SCREEN_WIDTH, SIDEBAR_WIDTH, FULL_SCREEN_WIDTH
)
import numpy as np
import pygame 

class TetrisEnv:

    def __init__(self, board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT, render_mode='human'):
        # 기존 초기화 로직 유지
        self.BOARD_WIDTH = board_width
        self.BOARD_HEIGHT = board_height
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        if self.render_mode == 'human':
            self._init_render()
            
        self.reset()

    # --- TetrisEnv.py 파일 내 TetrisEnv 클래스 내부 ---

    def _generate_new_bag(self):
        """7가지 테트로미노 인덱스(0-6)로 이루어진 새로운 Bag을 생성하고 무작위로 섞습니다."""
        # 7개의 블록 인덱스 (0부터 6까지)
        new_bag = list(range(len(TETROMINOS)))
        random.shuffle(new_bag)
        return new_bag

    def _get_next_block_index(self):
        """Bag에서 다음 블록 인덱스를 가져오고, Bag이 비면 새 Bag을 생성합니다."""
        if not self.bag:
            self.bag = self._generate_new_bag()
        
        # Bag의 맨 앞 블록을 꺼냅니다.
        return self.bag.pop(0)

    def _fill_next_blocks_queue(self, queue_size=3):
        """Next Block 큐를 지정된 크기로 채웁니다."""
        # Next Block 큐는 보통 3~6개로 설정됩니다.
        queue = []
        for _ in range(queue_size):
            queue.append(self._get_next_block_index())
        return queue

    def _new_block(self):
        """Next 큐에서 블록을 꺼내 Current Block으로 만들고, 큐를 다시 채웁니다."""
    
        # 큐가 비어있다면 오류 방지를 위해 채웁니다. (일반적으로는 비지 않음)
        if not self.next_blocks:
            self.next_blocks = self._fill_next_blocks_queue()
        
        # 1. 큐에서 다음 블록 인덱스를 꺼냅니다.
        shape_index = self.next_blocks.pop(0)
    
        # 2. 큐에 새로운 블록을 하나 추가하여 채웁니다.
        self.next_blocks.append(self._get_next_block_index())
    
        # 3. 새로운 CurrentBlock 객체를 생성하여 반환합니다.
        return CurrentBlock(self.game_board, shape_index) # self.game_board는 TetrisEnv 내에서 정의된 보드 배열

    def _init_render(self):
        """Pygame 화면 및 시계를 초기화합니다."""
        if self.screen is None:
            # Tetris.py의 전역 상수를 사용합니다.
            pygame.init()
            self.screen = pygame.display.set_mode((FULL_SCREEN_WIDTH, BOARD_HEIGHT * SQUARE_SIZE))
            pygame.display.set_caption("DQN Tetris 학습 중")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def get_state(self):
        # ... (기존 get_state 로직 유지) ...
        heights = np.zeros(self.BOARD_WIDTH, dtype=int)
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                if self.game_board[y][x] != 0:
                    heights[x] = self.BOARD_HEIGHT - y 
                    break
        
        holes = 0
        for x in range(self.BOARD_WIDTH):
            found_block = False
            for y in range(self.BOARD_HEIGHT):
                if self.game_board[y][x] != 0:
                    found_block = True
                elif found_block and self.game_board[y][x] == 0:
                    holes += 1

        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
        max_height = np.max(heights) if heights.size > 0 else 0
        # 상태 벡터 순서: [MaxHeight, Holes, Bumpiness, Col_0_Height, ...]
        state_features = np.array([max_height, holes, bumpiness] + heights.tolist())
        
        return state_features
        
    def reset(self):
        self.game_board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)
        self.next_blocks = [generate_random_block_index() for _ in range(5)]
        self.held_block_index = -1
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        current_block_index = self.next_blocks.pop(0)
        # 💡 CurrentBlock 초기화 수정: CurrentBlock이 board와 index를 받도록 가정
        self.current_block = CurrentBlock(self.game_board, current_block_index) 
        self.next_blocks.append(generate_random_block_index())

        return self.get_state()

    def step(self, action):
        if self.game_over:
            # 게임 오버 상태에서 step을 밟으면 보상 0, 종료
            return self.get_state(), 0, True, {}

        rotation, final_x = action
    
        # 1. 블록 회전 및 위치 조정
        self.current_block.rotate_to(rotation)
        self.current_block.x = final_x
    
        # 2. 블록을 최종 위치(final_x)까지 하드 드롭
        final_y = get_ghost_y(self.game_board, self.current_block)
        self.current_block.y = final_y
    
        # 3. 블록 잠금 및 줄 제거
        # 현재 보드에 블록 고정
        for row_idx, row in enumerate(self.current_block.shape):
            for col_idx, cell in enumerate(row):
                if cell != 0:
                    self.game_board[self.current_block.y + row_idx][self.current_block.x + col_idx] = self.current_block.shape_index + 1
    
        # 줄 제거
        cleared_lines, new_board = clear_lines(self.game_board)
        self.game_board = new_board
    
        # 4. 다음 블록 생성
        current_block_index = self.next_blocks.pop(0)
        self.current_block = CurrentBlock(self.game_board, current_block_index)
        self.next_blocks.append(generate_random_block_index())

        # 5. 게임 오버 확인
        
        # 💡 조건 1: 새로운 블록이 움직일 수 없으면 게임 오버
        if not self.current_block.can_move(self.game_board, 0, 0):
            self.game_over = True
        
        # np.any(self.game_board[0])는 보드의 0번째 행(Y=0)에 0이 아닌 값(고정된 블록)이 있는지 검사합니다.
        if np.any(self.game_board[0]): 
             self.game_over = True
        
        # 6. 보상 계산 (새로운 상태를 얻은 후 계산)
        next_state = self.get_state()
    
        # 💡 _calculate_reward 호출 시 next_state를 인수로 전달
        reward = self._calculate_reward(cleared_lines, next_state) 
    
        self.lines_cleared += cleared_lines

        # 7. 게임 오버 페널티 적용 (가장 큰 페널티)
        if self.game_over:
            # 💡 게임 오버 페널티 -500 적용
            reward = -500 
        
        return next_state, reward, self.game_over, {}
        
    def _calculate_reward(self, lines_cleared, state_features):
    
        # 1. 줄 제거 보상 (R_line) - 0줄 제거 시 보상을 0으로 설정
        if lines_cleared == 1: R_line = 500
        elif lines_cleared == 2: R_line = 1000
        elif lines_cleared == 3: R_line = 2000
        elif lines_cleared == 4: R_line = 3000
        else: R_line = 1
    
        # 2. 보드 상태 페널티 (P_heuristics)
        max_height = state_features[0]
        holes = state_features[1]
        bumpiness = state_features[2]
    
        # --- 페널티 계수 설정 (조정) ---
        # MaxHeight 페널티를 강화하여 높이 상승을 강력히 억제합니다.
        ALPHA = 1.0   # MaxHeight 계수 (0.5 -> 1.0으로 강화)
        BETA = 0.5    # Holes 계수
        GAMMA = 0.2  # Bumpiness 계수
        TIME_PENALTY = 0.02 # 시간 페널티 (0.01 -> 0.02로 약간 강화)

        P_heuristics = (ALPHA * max_height) + (BETA * holes) + (GAMMA * bumpiness)
    
        # 총 보상 = R_line - P_heuristics - P_time
        # 줄 제거가 없으면 R_line=0이므로, reward는 음수가 됩니다.
        reward = R_line - P_heuristics - TIME_PENALTY
    
        return reward

    def render(self, score=0.0, step_count=0):
        """
        게임 화면을 렌더링하고, 점수와 스텝 수를 표시합니다.
        """
        if self.render_mode != 'human':
            return
        
        # TetrisEnv 내에서 pygame 상수를 사용한다고 가정
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
    
        # 1. 화면 초기화
        self.screen.fill(BLACK)
    
        # 2. 보드, 홀드 블록, 다음 블록 그리기
        # *주의: 이 함수들이 TetrisEnv가 접근할 수 있는 곳에 정의되어 있어야 합니다.
        # 예: from Tetris import draw_board, draw_held_block, draw_next_blocks
        draw_board(self.screen, self.game_board, self.current_block)
        draw_held_block(self.screen, self.held_block_index)
        draw_next_blocks(self.screen, self.next_blocks[:-1]) # 마지막 큐는 제외 (선택적)
    
        # 3. 💡 점수 및 스텝 수 표시 로직 추가 (오류 해결 및 기능 추가)
    
        # TetrisEnv가 초기화될 때 폰트가 초기화되었다고 가정합니다 (pygame.font.init() 필요).
        try:
            font = pygame.font.Font(None, 30)
        except pygame.error:
            # 폰트 로딩 실패 시 임시 방편
            font = pygame.font.SysFont("Arial", 30)

        # 화면 너비와 보드 크기 상수를 TetrisEnv가 가지고 있다고 가정합니다.
        # SCREEN_WIDTH는 보드 영역 옆 사이드바 시작 지점입니다.
        # 예: self.BOARD_WIDTH * self.SQUARE_SIZE
    
        # 상수가 없다고 가정하고 임시 값 사용 (실제 TetrisEnv 파일에서 정확한 상수로 대체하세요)
        SCREEN_WIDTH_START = 300 # 보드 옆 사이드바가 시작되는 대략적인 X 좌표
    
        # 4. 화면 업데이트
        pygame.display.flip()
        self.clock.tick(5) # 초당 5프레임으로 제한

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def get_possible_actions(self):
        # ... (기존 get_possible_actions 로직 유지) ...
        actions = [] 
        current_shape_index = self.current_block.shape_index
        max_rotations = len(TETROMINOS[current_shape_index])
        
        for rot in range(max_rotations):
            temp_shape = TETROMINOS[current_shape_index][rot]
            block_width = len(temp_shape[0])

            for x in range(self.BOARD_WIDTH - block_width + 1):
                actions.append((rot, x)) 
                
        return actions