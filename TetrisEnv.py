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
import random

class TetrisEnv:

    def __init__(self, board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT, render_mode='human'):
        self.BOARD_WIDTH = board_width
        self.BOARD_HEIGHT = board_height
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 렌더링 모드일 때만 화면 초기화
        if self.render_mode == 'human':
            self._init_render()
            
        self.reset()

    def _generate_new_bag(self):
        new_bag = list(range(len(TETROMINOS)))
        random.shuffle(new_bag)
        return new_bag

    def _get_next_block_index(self):
        if not hasattr(self, 'bag') or not self.bag:
            self.bag = self._generate_new_bag()
        return self.bag.pop(0)

    def _fill_next_blocks_queue(self, queue_size=3):
        queue = []
        for _ in range(queue_size):
            queue.append(self._get_next_block_index())
        return queue

    def _new_block(self):
        if not self.next_blocks:
            self.next_blocks = self._fill_next_blocks_queue()
        shape_index = self.next_blocks.pop(0)
        self.next_blocks.append(self._get_next_block_index())
        return CurrentBlock(self.game_board, shape_index)

    def _init_render(self):
        if self.render_mode != 'human':
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((FULL_SCREEN_WIDTH, BOARD_HEIGHT * SQUARE_SIZE))
            pygame.display.set_caption("DQN Tetris 학습 중")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def get_state(self):
        # 1. 각 열의 높이(Heights) 계산
        heights = np.zeros(self.BOARD_WIDTH, dtype=int)
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                if self.game_board[y][x] != 0:
                    heights[x] = self.BOARD_HEIGHT - y 
                    break
        
        # 2. 구멍(Holes) 개수 계산
        holes = 0
        for x in range(self.BOARD_WIDTH):
            found_block = False
            for y in range(self.BOARD_HEIGHT):
                if self.game_board[y][x] != 0:
                    found_block = True
                elif found_block and self.game_board[y][x] == 0:
                    holes += 1

        # 3. 거칠기(Bumpiness) 및 최대 높이, 총 높이 계산
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
        max_height = np.max(heights) if heights.size > 0 else 0
        aggregate_height = np.sum(heights) # [추가] 총 높이 합

        # 4. 데이터 정규화 (Normalization)
        norm_max_height = max_height / self.BOARD_HEIGHT
        norm_holes = holes / (self.BOARD_WIDTH * self.BOARD_HEIGHT / 2) 
        norm_bumpiness = bumpiness / (self.BOARD_HEIGHT * self.BOARD_WIDTH)
        norm_aggregate = aggregate_height / (self.BOARD_HEIGHT * self.BOARD_WIDTH) # [추가]
        norm_heights = heights / self.BOARD_HEIGHT

        # 5. 상태 벡터 반환 (크기: 14)
        # 순서: [MaxHeight, Holes, Bumpiness, AggregateHeight, Col_0 ... Col_9]
        state_features = np.array([norm_max_height, norm_holes, norm_bumpiness, norm_aggregate] + norm_heights.tolist())
        
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
        self.current_block = CurrentBlock(self.game_board, current_block_index) 
        self.next_blocks.append(generate_random_block_index())

        # [중요] 초기 상태 저장 (Step에서 변화량 계산용)
        initial_state = self.get_state()
        self.prev_state_features = initial_state

        return initial_state

    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True, {}

        rotation, final_x = action
    
        # 1. 블록 회전 및 위치 조정
        self.current_block.rotate_to(rotation)
        self.current_block.x = final_x
    
        # 2. 하드 드롭
        final_y = get_ghost_y(self.game_board, self.current_block)
        self.current_block.y = final_y
    
        # 3. 블록 잠금
        for row_idx, row in enumerate(self.current_block.shape):
            for col_idx, cell in enumerate(row):
                if cell != 0:
                    self.game_board[self.current_block.y + row_idx][self.current_block.x + col_idx] = self.current_block.shape_index + 1
    
        # 4. 줄 제거
        cleared_lines, new_board = clear_lines(self.game_board)
        self.game_board = new_board
    
        # 5. 다음 블록 생성
        current_block_index = self.next_blocks.pop(0)
        self.current_block = CurrentBlock(self.game_board, current_block_index)
        self.next_blocks.append(generate_random_block_index())

        # 6. 게임 오버 확인 (생성되자마자 움직일 수 없거나 맨 윗줄 침범 시)
        if not self.current_block.can_move(self.game_board, 0, 0):
            self.game_over = True
        if np.any(self.game_board[0]): 
             self.game_over = True
        
        # 7. 보상 계산 (Delta 방식)
        next_state = self.get_state()
        
        # 이전 상태와 현재 상태의 차이를 기반으로 보상 계산
        reward = self._calculate_reward(cleared_lines, self.prev_state_features, next_state)
        
        self.lines_cleared += cleared_lines

        # [중요] 현재 상태를 이전 상태 변수에 업데이트
        self.prev_state_features = next_state

        # 8. 게임 오버 페널티
        if self.game_over:
            reward = -100 # 적당한 페널티
        
        return next_state, reward, self.game_over, {}
        
    def _calculate_reward(self, lines_cleared, prev_state, current_state):
        """
        변화량(Delta)을 기반으로 보상을 계산합니다.
        State 순서: [MaxHeight, Holes, Bumpiness, AggregateHeight, ...]
        """
        
        # 1. 줄 제거 보상 (기하급수적)
        if lines_cleared == 0: reward_lines = 0.0
        elif lines_cleared == 1: reward_lines = 10.0 # 기존 1.0 -> 10.0
        elif lines_cleared == 2: reward_lines = 30.0 # 기존 3.0 -> 30.0
        elif lines_cleared == 3: reward_lines = 60.0 # 기존 6.0 -> 60.0
        elif lines_cleared == 4: reward_lines = 100.0 # 기존 10.0 -> 100.0 (Tetris!)
        else: reward_lines = 0.0

        # 2. 상태 변화량 계산
        diff_height = current_state[0] - prev_state[0]    # Max Height 변화
        diff_holes = current_state[1] - prev_state[1]     # Holes 변화
        diff_bumpiness = current_state[2] - prev_state[2] # Bumpiness 변화
        diff_aggregate = current_state[3] - prev_state[3] # Aggregate Height 변화

        # 3. 가중치 설정 (정규화된 값이므로 가중치를 크게 설정)
        W_HEIGHT = -15.0      # 높이 증가 억제
        W_HOLES = -60.0       # 구멍 생성 강력 억제 (가장 중요)
        W_BUMPINESS = -5.0    # 평평하게 유지
        W_AGGREGATE = -2.0    # 전체 블록 높이 억제
        
        # 4. 최종 보상 합산
        reward = reward_lines \
                 + (W_HEIGHT * diff_height) \
                 + (W_HOLES * diff_holes) \
                 + (W_BUMPINESS * diff_bumpiness) \
                 + (W_AGGREGATE * diff_aggregate) \
                 + 0.01  # Step Reward (생존 보상)

        return reward

    def render(self, score=0.0, step_count=0):
        if self.render_mode != 'human':
            return
        
        self.screen.fill(BLACK)
        draw_board(self.screen, self.game_board, self.current_block)
        draw_held_block(self.screen, self.held_block_index)
        draw_next_blocks(self.screen, self.next_blocks[:-1])
        
        # 간단한 폰트 처리
        try:
            font = pygame.font.Font(None, 30)
        except:
            font = pygame.font.SysFont("Arial", 30)

        pygame.display.flip()
        self.clock.tick(10) # 렌더링 속도 조절

    def close(self):
        if self.screen is not None:
            pygame.quit()

    def get_possible_actions(self):
        actions = [] 
        current_shape_index = self.current_block.shape_index
        max_rotations = len(TETROMINOS[current_shape_index])
        
        for rot in range(max_rotations):
            temp_shape = TETROMINOS[current_shape_index][rot]
            block_width = len(temp_shape[0])

            for x in range(self.BOARD_WIDTH - block_width + 1):
                actions.append((rot, x)) 
                
        return actions