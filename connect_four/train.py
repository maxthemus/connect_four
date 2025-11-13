import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game import create_board, board_to_tensor, apply_move, print_board
from model import ConnectFourModel
import random
from collections import deque


def check_winner(board):
    """
    Check if there's a winner on the board.

    Returns:
        1 if player 1 wins, -1 if player 2 wins, 0 if no winner
    """
    rows, cols = board.shape

    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if abs(window.sum()) == 4:
                return int(np.sign(window.sum()))

    # Check vertical
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if abs(window.sum()) == 4:
                return int(np.sign(window.sum()))

    # Check diagonal (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if abs(sum(window)) == 4:
                return int(np.sign(sum(window)))

    # Check diagonal (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if abs(sum(window)) == 4:
                return int(np.sign(sum(window)))

    return 0


def check_winner_type(board):
    """
    Check if there's a winner and return the win type.

    Returns:
        tuple: (winner, win_type) where winner is 1/-1/0 and 
               win_type is 'horizontal', 'vertical', 'diagonal_right', 'diagonal_left', or None
    """
    rows, cols = board.shape

    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if abs(window.sum()) == 4:
                return int(np.sign(window.sum())), 'horizontal'

    # Check vertical
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if abs(window.sum()) == 4:
                return int(np.sign(window.sum())), 'vertical'

    # Check diagonal (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if abs(sum(window)) == 4:
                return int(np.sign(sum(window))), 'diagonal_right'

    # Check diagonal (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if abs(sum(window)) == 4:
                return int(np.sign(sum(window))), 'diagonal_left'

    return 0, None


def get_valid_moves(board):
    """Get list of valid column indices where a piece can be placed."""
    cols = board.shape[1]
    return [col for col in range(cols) if board[0, col] == 0]


def check_threat(board, player):
    """
    Check if there's a 3-in-a-row threat for the given player.
    Returns True if player has 3 in a row with an empty 4th spot that is playable.

    Args:
        board: Current board state
        player: Player to check threats for (1 or -1)

    Returns:
        True if player has a threat (3 in a row with playable 4th spot), False otherwise
    """
    rows, cols = board.shape

    def is_playable(row, col):
        """Check if a position is playable (empty and either bottom row or has support below)"""
        if board[row, col] != 0:
            return False
        # Bottom row is always playable
        if row == rows - 1:
            return True
        # Otherwise needs support below
        return board[row + 1, col] != 0

    # Check horizontal threats
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row, col + i] == 0:
                        if is_playable(row, col + i):
                            return True

    # Check vertical threats
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col] == 0:
                        if is_playable(row + i, col):
                            return True

    # Check diagonal threats (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col + i] == 0:
                        if is_playable(row + i, col + i):
                            return True

    # Check diagonal threats (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col - i] == 0:
                        if is_playable(row + i, col - i):
                            return True

    return False


def count_threats(board, player):
    """
    Count the number of 3-in-a-row threats for the given player.
    A threat is 3 in a row with an empty 4th spot that is playable.

    Args:
        board: Current board state
        player: Player to check threats for (1 or -1)

    Returns:
        Number of threats found
    """
    rows, cols = board.shape
    threat_count = 0

    def is_playable(row, col):
        """Check if a position is playable (empty and either bottom row or has support below)"""
        if board[row, col] != 0:
            return False
        if row == rows - 1:
            return True
        return board[row + 1, col] != 0

    # Check horizontal threats
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                for i in range(4):
                    if board[row, col + i] == 0:
                        if is_playable(row, col + i):
                            threat_count += 1
                            break

    # Check vertical threats
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                for i in range(4):
                    if board[row + i, col] == 0:
                        if is_playable(row + i, col):
                            threat_count += 1
                            break

    # Check diagonal threats (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                for i in range(4):
                    if board[row + i, col + i] == 0:
                        if is_playable(row + i, col + i):
                            threat_count += 1
                            break

    # Check diagonal threats (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                for i in range(4):
                    if board[row + i, col - i] == 0:
                        if is_playable(row + i, col - i):
                            threat_count += 1
                            break

    return threat_count


def count_two_in_a_row(board, player):
    """
    Count the number of 2-in-a-row opportunities for the given player.
    An opportunity is 2 in a row with at least one playable empty spot adjacent.

    Args:
        board: Current board state
        player: Player to check for (1 or -1)

    Returns:
        Number of 2-in-a-row opportunities found
    """
    rows, cols = board.shape
    count = 0

    def is_playable(row, col):
        """Check if a position is playable"""
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
        if board[row, col] != 0:
            return False
        if row == rows - 1:
            return True
        return board[row + 1, col] != 0

    # Check horizontal 2-in-a-row (with room to extend)
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if np.sum(window == player) == 2 and np.sum(window == 0) == 2:
                # Check if any empty spot is playable
                has_playable = False
                for i in range(4):
                    if board[row, col + i] == 0 and is_playable(row, col + i):
                        has_playable = True
                        break
                if has_playable:
                    count += 1

    # Check vertical 2-in-a-row
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if np.sum(window == player) == 2 and np.sum(window == 0) == 2:
                has_playable = False
                for i in range(4):
                    if board[row + i, col] == 0 and is_playable(row + i, col):
                        has_playable = True
                        break
                if has_playable:
                    count += 1

    # Check diagonal 2-in-a-row (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 2
                and sum(1 for x in window if x == 0) == 2
            ):
                has_playable = False
                for i in range(4):
                    if board[row + i, col + i] == 0 and is_playable(row + i, col + i):
                        has_playable = True
                        break
                if has_playable:
                    count += 1

    # Check diagonal 2-in-a-row (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 2
                and sum(1 for x in window if x == 0) == 2
            ):
                has_playable = False
                for i in range(4):
                    if board[row + i, col - i] == 0 and is_playable(row + i, col - i):
                        has_playable = True
                        break
                if has_playable:
                    count += 1

    return count


def get_blocking_columns(board, player):
    """
    Get all columns that would block a 4-in-a-row for the given player.
    
    Args:
        board: Current board state
        player: Player to check blocking moves for (1 or -1)
    
    Returns:
        List of column indices that would block a potential 4-in-a-row
    """
    rows, cols = board.shape
    blocking_cols = set()
    
    def is_playable(row, col):
        """Check if a position is playable (empty and either bottom row or has support below)"""
        if board[row, col] != 0:
            return False
        # Bottom row is always playable
        if row == rows - 1:
            return True
        # Otherwise needs support below
        return board[row + 1, col] != 0
    
    # Check horizontal windows
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row, col + i] == 0:
                        if is_playable(row, col + i):
                            blocking_cols.add(col + i)
    
    # Check vertical windows
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col] == 0:
                        if is_playable(row + i, col):
                            blocking_cols.add(col)
    
    # Check diagonal windows (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col + i] == 0:
                        if is_playable(row + i, col + i):
                            blocking_cols.add(col + i)
    
    # Check diagonal windows (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                # Find which position is empty and check if it's playable
                for i in range(4):
                    if board[row + i, col - i] == 0:
                        if is_playable(row + i, col - i):
                            blocking_cols.add(col - i)
    
    return sorted(list(blocking_cols))


def count_threes(board, player):
    """Count number of *open* 3-in-a-row windows for player (possible to become 4)."""
    rows, cols = board.shape
    count = 0

    # Horizontal windows of length 4 that contain exactly 3 player's pieces and 1 empty
    for r in range(rows):
        for c in range(cols - 3):
            w = board[r, c : c + 4]
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            w = board[r : r + 4, c]
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1

    # Diagonals (down-right and down-left)
    for r in range(rows - 3):
        for c in range(cols - 3):
            w = [board[r + i, c + i] for i in range(4)]
            if sum(1 for x in w if x == player) == 3 and sum(1 for x in w if x == 0) == 1:
                count += 1
        for c in range(3, cols):
            w = [board[r + i, c - i] for i in range(4)]
            if sum(1 for x in w if x == player) == 3 and sum(1 for x in w if x == 0) == 1:
                count += 1

    return count

def calculate_reward(board, prev_board, col, player, winner, is_done):
    reward = 0.0

    # Terminal rewards
    if is_done:
        if winner == player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return -1.0

    # Small step penalty to discourage meaningless moves
    reward -= 0.01

    opponent = -player

    # 1. Prevent opponent from winning next move (HIGHEST PRIORITY)
    # Look ahead one move: if opponent could win next turn, heavily reward blocking it
    blocked_win = False
    for c in get_valid_moves(board):
        test_board = board.copy()
        apply_move(test_board, c, opponent)
        if check_winner(test_board) == opponent:
            if c == col:
                reward += 2.0  # blocked imminent win - CRITICAL!
                blocked_win = True
            else:
                reward -= 0.5  # didn't block imminent win - PUNISH HARDER

    # 2. Reward for reducing opponent's 3-in-a-row threats
    if not blocked_win:
        prev_opponent_threats = count_threats(prev_board, opponent)
        curr_opponent_threats = count_threats(board, opponent)
        
        # Reward for eliminating opponent threats
        if prev_opponent_threats > curr_opponent_threats:
            threats_stopped = prev_opponent_threats - curr_opponent_threats
            reward += 0.6 * threats_stopped  # Reward per threat stopped
        
        # Legacy check_threat bonus (keep for backward compatibility)
        if check_threat(prev_board, opponent) and not check_threat(board, opponent):
            reward += 0.2  # Additional small bonus

    # 3. Create your own threats
    prev_player_threats = count_threats(prev_board, player)
    curr_player_threats = count_threats(board, player)
    
    if curr_player_threats > prev_player_threats:
        new_threats = curr_player_threats - prev_player_threats
        reward += 0.4 * new_threats  # Reward per new threat created

    # 4. Early intervention: Reward blocking opponent's 2-in-a-row (proactive defense)
    prev_opponent_two = count_two_in_a_row(prev_board, opponent)
    curr_opponent_two = count_two_in_a_row(board, opponent)
    
    if prev_opponent_two > curr_opponent_two:
        two_stopped = prev_opponent_two - curr_opponent_two
        reward += 0.3 * two_stopped  # Reward early blocking
    
    # 5. Reward creating your own 2-in-a-row (building offensive positions)
    prev_player_two = count_two_in_a_row(prev_board, player)
    curr_player_two = count_two_in_a_row(board, player)
    
    if curr_player_two > prev_player_two:
        two_created = curr_player_two - prev_player_two
        reward += 0.2 * two_created  # Reward building positions

    # 6. Encourage central columns
    center_col = board.shape[1] // 2
    if col == center_col:
        reward += 0.05

    return reward



def check_winner_after_move(board, player, col):
    tmp = board.copy()
    if not apply_move(tmp, col, player):
        return False
    return check_winner(tmp) == player


def select_move(model, board, epsilon, valid_moves, device='cpu'):
    """Select a move using epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return np.random.choice(valid_moves)
    
    model.eval()
    with torch.no_grad():
        # Convert board to tensor and flatten to 1D
        state = board_to_tensor(board).flatten().unsqueeze(0).to(device)
        q_values = model(state).cpu().numpy()[0]
    model.train()
    
    # Mask invalid moves
    q_values_masked = q_values.copy()
    for col in range(7):
        if col not in valid_moves:
            q_values_masked[col] = -np.inf
    
    return np.argmax(q_values_masked)

def select_move_blocking(model, board, player, epsilon, valid_moves, device='cpu'):
    """Select a move with blocking logic."""
    opponent = -player
    
    # 1. Check if we can win immediately
    for col in valid_moves:
        test_board = board.copy()
        row = np.where(test_board[:, col] == 0)[0]
        if len(row) > 0:
            test_board[row[-1], col] = player
            if check_winner(test_board) == player:
                return col
    
    # 2. Check if opponent has a threat and if we can block it
    if check_threat(board, opponent):
        blocking_cols = get_blocking_columns(board, opponent)
        valid_blocking = [col for col in blocking_cols if col in valid_moves]
        
        if valid_blocking:
            # If multiple blocking moves, prefer ones that create our own threats
            best_block = valid_blocking[0]
            for col in valid_blocking:
                test_board = board.copy()
                row = np.where(test_board[:, col] == 0)[0]
                if len(row) > 0:
                    test_board[row[-1], col] = player
                    if check_threat(test_board, player):
                        best_block = col
                        break
            return best_block
    
    # 3. Use model for normal play
    model.eval()
    with torch.no_grad():
        # Convert board to tensor and flatten to 1D
        state = board_to_tensor(board).flatten().unsqueeze(0).to(device)
        q_values = model(state).cpu().numpy()[0]
    model.train()
    
    # Apply epsilon-greedy
    if np.random.random() < epsilon:
        return np.random.choice(valid_moves)
    
    # Mask invalid moves
    q_values_masked = q_values.copy()
    for col in range(7):
        if col not in valid_moves:
            q_values_masked[col] = -np.inf
    
    return np.argmax(q_values_masked)
def play_game(
    model1,
    model2,
    epsilon1=0.1,
    epsilon2=0.1,
    verbose=False,
    starting_player=1,
    random_opponent=False,
    use_blocking_opponent=False,
):
    """
    Play a single game between two models.

    Args:
        starting_player: Which player goes first (1 or -1)
        random_opponent: If True, player 2 plays randomly
        use_blocking_opponent: If True, player 2 uses blocking strategy

    Returns:
        tuple: (game_history, winner, win_type) where game_history is list of tuples,
               winner is 1/-1/0, and win_type is the type of win or None
    """
    board = create_board()
    game_history = []
    current_player = starting_player
    models = {1: model1, -1: model2}
    epsilons = {1: epsilon1, -1: epsilon2}

    for move_count in range(42):  # Max 42 moves in Connect 4
        # Get current state (before move)
        state = board.copy()

        # Select and apply move
        if random_opponent and current_player == -1:
            col = random.choice(get_valid_moves(board))
        elif use_blocking_opponent and current_player == -1:
            col = select_move_blocking(
                models[current_player], board, current_player, epsilons[current_player], get_valid_moves(board)
            )
        else:
            col = select_move(
                models[current_player], board, epsilons[current_player], get_valid_moves(board)
            )

        if col is None:  # Board is full
            # Record draw for last move with updated reward
            if game_history:
                prev_state, prev_col, _, prev_next_state, _, prev_player = game_history[
                    -1
                ]
                reward = calculate_reward(
                    board, prev_state, prev_col, prev_player, 0, True
                )
                game_history[-1] = (
                    prev_state,
                    prev_col,
                    reward,
                    board.copy(),
                    True,
                    prev_player,
                )
            return game_history, 0, None

        prev_board = state.copy()
        success, row = apply_move(board, col, current_player)

        if not success:
            reward = -0.5
            game_history.append((state, col, reward, board.copy(), False, current_player))
            return game_history, 0, None

        # Check for winner with type
        winner, win_type = check_winner_type(board)
        is_done = winner != 0

        # Calculate reward using the new reward function
        reward = calculate_reward(
            board, prev_board, col, current_player, winner, is_done
        )
        game_history.append((state, col, reward, board.copy(), is_done, current_player))

        if is_done:
            if verbose:
                if winner == current_player:
                    print(f"\nPlayer {current_player} wins!")
                print_board(board)
            return game_history, winner, win_type

        # Switch player
        current_player *= -1
    else:
        # Draw - no winner after 42 moves
        if game_history:
            prev_state, prev_col, _, prev_next_state, _, prev_player = game_history[-1]
            reward = calculate_reward(board, prev_state, prev_col, prev_player, 0, True)
            game_history[-1] = (
                prev_state,
                prev_col,
                reward,
                board.copy(),
                True,
                prev_player,
            )

        if verbose:
            print("\nGame ended in a draw!")
            print_board(board)

    return game_history, 0, None


def train_step(model, target_model, optimizer, batch, gamma=0.99, device='cpu'):
    model.train()
    # Unpack batch
    states, actions, rewards, next_states, dones, players = zip(*batch)

    # Convert boards to tensors and flatten to 1D (batch_size, 42)
    states_tensor = torch.stack([board_to_tensor(s * p).flatten() for s, p in zip(states, players)]).to(device)
    next_states_tensor = torch.stack([board_to_tensor(ns * p).flatten() for ns, p in zip(next_states, players)]).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)

    # Current Q for taken actions
    q_out = model(states_tensor)                      # (batch, 7)
    current_q_values = q_out.gather(1, actions_tensor.unsqueeze(1)).squeeze()

    # Double DQN target calculation:
    # 1) online model selects best next action
    with torch.no_grad():
        next_q_online = model(next_states_tensor)            # (batch,7)
        # We still need to mask invalid actions in next states before argmax
        masks = []
        for ns in next_states:  # ns is raw numpy board (6x7)
            valid = get_valid_moves(ns)
            m = torch.full((7,), float('-inf'))
            m[valid] = 0.0
            masks.append(m)
        masks = torch.stack(masks).to(device)                # (batch,7)
        masked_next_q_online = next_q_online + masks
        next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)  # (batch,1)

        # 2) evaluate selected actions with the target network
        next_q_target = target_model(next_states_tensor)     # (batch,7)
        next_q_values = next_q_target.gather(1, next_actions).squeeze()  # (batch,)

        target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

    # Loss (Huber/SmoothL1)
    loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()




def train(
    num_episodes=1000,
    batch_size=32,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    target_update_freq=100,
    use_cuda=True,
):
    """
    Train the model using DQN with self-play and target network.

    Args:
        num_episodes: Number of games to play
        batch_size: Size of training batches
        learning_rate: Learning rate for optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Decay rate for epsilon
        target_update_freq: How often to update target network (in episodes)
        use_cuda: If True, use CUDA for GPU acceleration (if available)
    """
    # Set up device
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Attempting to use CUDA GPU: {torch.cuda.get_device_name(0)}")
        
        # Test if CUDA actually works with a simple operation
        try:
            test_tensor = torch.zeros(1).to(device)
            _ = test_tensor + 1
            print(f"✓ CUDA is working properly")
        except RuntimeError as e:
            print(f"✗ CUDA error detected: {e}")
            print(f"Falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        if use_cuda:
            print("CUDA requested but not available")
        print("Using CPU")
    
    # Create main model and target model
    model = ConnectFourModel().to(device)
    target_model = ConnectFourModel().to(device)
    target_model.load_state_dict(model.state_dict())  # Initialize with same weights
    target_model.eval()  # Target network is not trained directly

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=10000)
    epsilon = epsilon_start

    wins = {1: 0, -1: 0, 0: 0}  # Track wins, losses, draws
    win_types = {'horizontal': 0, 'vertical': 0, 'diagonal_right': 0, 'diagonal_left': 0}  # Track win types
    model_as_p1_wins = 0  # Track model wins as Player 1
    model_as_p2_wins = 0  # Track model wins as Player 2
    model_as_p1_losses = 0  # Track model losses as Player 1
    model_as_p2_losses = 0  # Track model losses as Player 2
    total_model_games = 0
    game_lengths = []  # Track number of turns per game

    print("Starting DQN self-play training...")
    print(f"Target network will update every {target_update_freq} episodes\n")

    for episode in range(num_episodes):
        # Mix different opponent types for better training
        opponent_type = random.random()
        random_opponent = opponent_type < 0.15  # 15% random
        use_blocking_opponent = 0.15 <= opponent_type < 0.40  # 25% blocking opponent
        # 60% normal opponent (target network without special strategies)

        # Alternate which player the main model controls
        # Model plays as Player 1 (goes first) in even episodes
        # Model plays as Player 2 (goes second) in odd episodes
        if episode % 2 == 0:
            # Main model is Player 1, target is Player 2
            game_history, winner, win_type = play_game(
                model,
                target_model,
                epsilon,  # main model explores
                0.0,      # target doesn't explore
                verbose=False,
                starting_player=1,  # P1 always goes first
                random_opponent=random_opponent,
                use_blocking_opponent=use_blocking_opponent,
            )
            model_position = 1
        else:
            # Target is Player 1, main model is Player 2
            game_history, winner, win_type = play_game(
                target_model,
                model,
                0.0,      # target doesn't explore
                epsilon,  # main model explores
                verbose=False,
                starting_player=1,  # P1 always goes first
                random_opponent=False,  # Don't make P1 random when it's the target
                use_blocking_opponent=False,  # Don't give P1 blocking advantage
            )
            model_position = -1

        # Add to replay buffer
        replay_buffer.extend(game_history)

        # Track game length (number of moves)
        game_lengths.append(len(game_history))

        # Track statistics
        wins[winner] += 1
        if win_type:
            win_types[win_type] += 1
        
        # Track model performance in both positions
        total_model_games += 1
        if model_position == 1:
            # Model was Player 1
            if winner == 1:
                model_as_p1_wins += 1
            elif winner == -1:
                model_as_p1_losses += 1
        else:
            # Model was Player 2
            if winner == -1:
                model_as_p2_wins += 1
            elif winner == 1:
                model_as_p2_losses += 1

        # Train if we have enough samples
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            loss = train_step(model, target_model, optimizer, batch, gamma, device)
        else:
            loss = 0.0

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"  >>> Target network updated at episode {episode + 1}")

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 100 == 0:
            total_games = sum(wins.values())
            total_wins = sum(win_types.values())
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(
                f"  Win rates - P1: {wins[1]/total_games:.2%}, P2: {wins[-1]/total_games:.2%}, Draw: {wins[0]/total_games:.2%}"
            )
            # Show model performance in both positions
            p1_games = total_model_games // 2 + (total_model_games % 2)
            p2_games = total_model_games // 2
            if p1_games > 0:
                p1_draws = p1_games - model_as_p1_wins - model_as_p1_losses
                print(f"  Model as P1: {model_as_p1_wins}W-{model_as_p1_losses}L-{p1_draws}D ({model_as_p1_wins/p1_games:.1%} wins, {model_as_p1_losses/p1_games:.1%} losses)")
            if p2_games > 0:
                p2_draws = p2_games - model_as_p2_wins - model_as_p2_losses
                print(f"  Model as P2: {model_as_p2_wins}W-{model_as_p2_losses}L-{p2_draws}D ({model_as_p2_wins/p2_games:.1%} wins, {model_as_p2_losses/p2_games:.1%} losses)")
            # Show game length statistics
            if game_lengths:
                min_length = min(game_lengths)
                max_length = max(game_lengths)
                avg_length = sum(game_lengths) / len(game_lengths)
                print(f"  Game length: min={min_length}, max={max_length}, avg={avg_length:.1f} moves")
            if total_wins > 0:
                print(f"  Win types:")
                print(f"    Horizontal:     {win_types['horizontal']:3d} ({win_types['horizontal']/total_wins:.1%})")
                print(f"    Vertical:       {win_types['vertical']:3d} ({win_types['vertical']/total_wins:.1%})")
                print(f"    Diagonal Right: {win_types['diagonal_right']:3d} ({win_types['diagonal_right']/total_wins:.1%})")
                print(f"    Diagonal Left:  {win_types['diagonal_left']:3d} ({win_types['diagonal_left']/total_wins:.1%})")
            print(f"  Replay buffer size: {len(replay_buffer)}")
            wins = {1: 0, -1: 0, 0: 0}  # Reset stats
            win_types = {'horizontal': 0, 'vertical': 0, 'diagonal_right': 0, 'diagonal_left': 0}  # Reset win types
            model_as_p1_wins = 0
            model_as_p2_wins = 0
            model_as_p1_losses = 0
            model_as_p2_losses = 0
            total_model_games = 0
            game_lengths = []  # Reset game lengths

    print("\nTraining complete!")
    return model


def play_human_vs_model(model, use_blocking=False, verbose=False):
    """
    Play a game against the trained model.
    
    Args:
        model: The trained model to play against
        use_blocking: If True, use the blocking strategy for AI
        verbose: If True, show AI's decision-making process
    """
    board = create_board()
    current_player = 1  # Human is player 1

    print("\nYou are X (Player 1), AI is O (Player -1)")
    if use_blocking:
        print("AI is using blocking strategy")
    if verbose:
        print("Verbose mode: ON\n")
    print_board(board)

    while True:
        if current_player == 1:
            # Human move
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                print("Board is full! Draw!")
                break

            # Show if AI has threats
            if verbose:
                ai_threats = get_blocking_columns(board, -1)
                if ai_threats:
                    print(f"[Info] AI threatens to win in columns: {ai_threats}")
                human_threats = get_blocking_columns(board, 1)
                if human_threats:
                    print(f"[Info] You threaten to win in columns: {human_threats}")

            while True:
                try:
                    col = int(input(f"\nYour turn! Enter column (0-6): "))
                    if col in valid_moves:
                        break
                    else:
                        print("Invalid move! Column is full or out of range.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
        else:
            # AI move
            if use_blocking:
                col = select_move_blocking(model, board, current_player, epsilon=0.0, valid_moves=get_valid_moves(board))
            else:
                col = select_move(model, board, current_player, epsilon=0.0, valid_moves=get_valid_moves(board))
            if col is None:
                print("Board is full! Draw!")
                break
            print(f"\nAI plays column {col}")

        apply_move(board, col, current_player)
        print_board(board)

        winner = check_winner(board)
        if winner != 0:
            if winner == 1:
                print("\nYou win! Congratulations!")
            else:
                print("\nAI wins!")
            break

        current_player *= -1


# Add this diagnostic function to see how well the model actually plays

def evaluate_model(model, num_games=100, device='cpu'):
    """
    Evaluate model performance against different opponents.
    """
    model.eval()
    
    results = {
        'vs_random': {'wins': 0, 'losses': 0, 'draws': 0},
        'vs_blocking': {'wins': 0, 'losses': 0, 'draws': 0}
    }
    
    # Test against random opponent
    for _ in range(num_games):
        _, winner, _ = play_game(model, model, epsilon=0.0, random_opponent=True, device=device)
        if winner == 1:
            results['vs_random']['wins'] += 1
        elif winner == -1:
            results['vs_random']['losses'] += 1
        else:
            results['vs_random']['draws'] += 1
    
    # Test against blocking opponent
    for _ in range(num_games):
        _, winner, _ = play_game(model, model, epsilon=0.0, use_blocking_opponent=True, device=device)
        if winner == 1:
            results['vs_blocking']['wins'] += 1
        elif winner == -1:
            results['vs_blocking']['losses'] += 1
        else:
            results['vs_blocking']['draws'] += 1
    
    print(f"\nModel Evaluation ({num_games} games each):")
    print(f"  vs Random:   {results['vs_random']['wins']}W-{results['vs_random']['losses']}L-{results['vs_random']['draws']}D ({100*results['vs_random']['wins']/num_games:.1f}% win rate)")
    print(f"  vs Blocking: {results['vs_blocking']['wins']}W-{results['vs_blocking']['losses']}L-{results['vs_blocking']['draws']}D ({100*results['vs_blocking']['wins']/num_games:.1f}% win rate)")
    
    model.train()
    return results


if __name__ == "__main__":
    trained_model = None
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Train the model with DQN
    trained_model = train(
        num_episodes=5000,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=100,
        use_cuda=True,  # Enable CUDA if available
    )

    # Move model to CPU before saving for compatibility
    trained_model = trained_model.cpu()
    
    # Save the model
    torch.save(trained_model.state_dict(), "connect_four_model_v2.pth")
    print("Model saved to connect_four_model_v2.pth")
    if not trained_model:
        # Load the model if not trained in this session
        trained_model = ConnectFourModel()
        trained_model.load_state_dict(torch.load("connect_four_model_v2.pth", map_location='cpu'))
        trained_model.eval()
        print("Model loaded from connect_four_model_v2.pth")

    # Optional: Play against the model
    play_choice = input(
        "\nWould you like to play against the trained model? (yes/no): "
    )
    
    if play_choice.lower() in ["yes", "y"]:
        play_human_vs_model(trained_model, use_blocking=True, verbose=True)

    # Evaluate the trained model
    eval_results = evaluate_model(trained_model, num_games=100, device='cpu')