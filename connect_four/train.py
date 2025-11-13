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


def get_valid_moves(board):
    """Get list of valid column indices where a piece can be placed."""
    cols = board.shape[1]
    return [col for col in range(cols) if board[0, col] == 0]


def check_threat(board, player):
    """
    Check if there's a 3-in-a-row threat for the given player.
    Returns True if player has 3 in a row with an empty 4th spot.

    Args:
        board: Current board state
        player: Player to check threats for (1 or -1)

    Returns:
        True if player has a threat (3 in a row), False otherwise
    """
    rows, cols = board.shape

    # Check horizontal threats
    for row in range(rows):
        for col in range(cols - 3):
            window = board[row, col : col + 4]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                return True

    # Check vertical threats
    for row in range(rows - 3):
        for col in range(cols):
            window = board[row : row + 4, col]
            if np.sum(window == player) == 3 and np.sum(window == 0) == 1:
                return True

    # Check diagonal threats (down-right)
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row + i, col + i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                return True

    # Check diagonal threats (down-left)
    for row in range(rows - 3):
        for col in range(3, cols):
            window = [board[row + i, col - i] for i in range(4)]
            if (
                sum(1 for x in window if x == player) == 3
                and sum(1 for x in window if x == 0) == 1
            ):
                return True

    return False

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

    # 1. Block opponent threats (3 in a row)
    if check_threat(prev_board, opponent) and not check_threat(board, opponent):
        reward += 0.7

    # 2. Create your own threats
    if check_threat(board, player):
        reward += 0.3

    # 3. Prevent opponent from winning next move
    # Look ahead one move: if opponent could win next turn, reward blocking it
    for c in get_valid_moves(board):
        test_board = board.copy()
        apply_move(test_board, c, opponent)
        if check_winner(test_board) == opponent:
            if c == col:
                reward += 0.9  # blocked imminent win
            else:
                reward -= 0.2  # didn’t block imminent win

    # 4. Encourage central columns
    center_col = board.shape[1] // 2
    if col == center_col:
        reward += 0.05

    # 5. Optional: reward longer chains (2 or 3 in a row)
    # (could implement by scanning board for 2/3 consecutive pieces)
    # e.g. reward += 0.1 for each 2-in-a-row created

    return reward



def check_winner_after_move(board, player, col):
    tmp = board.copy()
    if not apply_move(tmp, col, player):
        return False
    return check_winner(tmp) == player


def select_move(model, board, player, epsilon=0.1):
    """
    Select a move using epsilon-greedy strategy.

    Args:
        model: The neural network model
        board: Current board state
        player: Current player (1 or -1)
        epsilon: Exploration rate

    Returns:
        Selected column index
    """
    valid_moves = get_valid_moves(board)

    if not valid_moves:
        return None

    # Exploration: random move
    if random.random() < epsilon:
        return random.choice(valid_moves)

    # Exploitation: use model
    model.eval()
    with torch.no_grad():
        # Prepare board for model (from player's perspective)
        board_input = board * player  # Flip perspective for player -1
        tensor_board = board_to_tensor(board_input).unsqueeze(0).unsqueeze(0)

        q_values = model(tensor_board).squeeze()

        # Mask invalid moves
        mask = torch.full((7,), float("-inf"))
        mask[valid_moves] = 0
        q_values = q_values + mask

        return q_values.argmax().item()


def play_game(
    model1,
    model2,
    epsilon1=0.1,
    epsilon2=0.1,
    verbose=False,
    starting_player=1,
    random_opponent=False,
):
    """
    Play a single game between two models.

    Args:
        starting_player: Which player goes first (1 or -1)

    Returns:
        List of (state, action, reward, next_state, done, player) tuples
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
        else:
            col = select_move(
                models[current_player], board, current_player, epsilons[current_player]
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
            break

        prev_board = state.copy()
        success, row = apply_move(board, col, current_player)

        if not success:
            reward = -0.5
            game_history.append((state, col, reward, board.copy(), False, current_player))
            break

        # Check for winner
        winner = check_winner(board)
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
            break

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

    return game_history, winner


def train_step(model, target_model, optimizer, batch, gamma=0.99, device='cpu'):
    model.train()
    # Unpack batch
    states, actions, rewards, next_states, dones, players = zip(*batch)

    # tensors (move to device)
    states_tensor = torch.stack([board_to_tensor(s * p).unsqueeze(0) for s, p in zip(states, players)]).to(device)
    next_states_tensor = torch.stack([board_to_tensor(ns * p).unsqueeze(0) for ns, p in zip(next_states, players)]).to(device)
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
    """
    # Create main model and target model
    model = ConnectFourModel()
    target_model = ConnectFourModel()
    target_model.load_state_dict(model.state_dict())  # Initialize with same weights
    target_model.eval()  # Target network is not trained directly

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=10000)
    epsilon = epsilon_start

    wins = {1: 0, -1: 0, 0: 0}  # Track wins, losses, draws

    print("Starting DQN self-play training...")
    print(f"Target network will update every {target_update_freq} episodes\n")

    for episode in range(num_episodes):
        # Alternate which player goes first to balance training
        random_opponent = random.random() < 0.3

        starting_player = 1 if episode % 2 == 0 else -1

        # Play against target network (frozen version provides stable opponent)
        game_history, winner = play_game(
            model,
            target_model,
            epsilon,
            0.0,
            verbose=False,
            starting_player=starting_player,
            random_opponent=random_opponent,
        )

        # Add to replay buffer
        replay_buffer.extend(game_history)

        # Track statistics
        wins[winner] += 1

        # Train if we have enough samples
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            loss = train_step(model, target_model, optimizer, batch, gamma)
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
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(
                f"  Win rates - P1: {wins[1]/total_games:.2%}, P2: {wins[-1]/total_games:.2%}, Draw: {wins[0]/total_games:.2%}"
            )
            print(f"  Replay buffer size: {len(replay_buffer)}")
            wins = {1: 0, -1: 0, 0: 0}  # Reset stats

    print("\nTraining complete!")
    return model


def play_human_vs_model(model):
    """
    Play a game against the trained model.
    """
    board = create_board()
    current_player = 1  # Human is player 1

    print("\nYou are X (Player 1), AI is O (Player -1)")
    print_board(board)

    while True:
        if current_player == 1:
            # Human move
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                print("Board is full! Draw!")
                break

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
            col = select_move(model, board, current_player, epsilon=0.0)
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


if __name__ == "__main__":
    # Train the model with DQN
    # trained_model = train(
    #     num_episodes=10000,
    #     batch_size=64,
    #     learning_rate=0.001,
    #     gamma=0.99,
    #     epsilon_start=1.0,
    #     epsilon_end=0.05,
    #     epsilon_decay=0.995,
    #     target_update_freq=100,
    # )

    # # Save the model
    # torch.save(trained_model.state_dict(), "connect_four_model.pth")
    # print("Model saved to connect_four_model.pth")

    # # Optional: Play against the model
    # play_choice = input(
    #     "\nWould you like to play against the trained model? (yes/no): "
    # )
    
    # # Load the trained model weights
    # trained_model.load_state_dict(torch.load("connect_four_model.pth"))
    # if play_choice.lower() in ["yes", "y"]:
    #     play_human_vs_model(trained_model)

    # load the trained model weights
    trained_model = ConnectFourModel()
    trained_model.load_state_dict(torch.load("connect_four_model.pth"))

    play_human_vs_model(trained_model)