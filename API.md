# Connect Four API Documentation

## Overview

This Django API provides an endpoint for making Connect Four game moves using a trained neural network model.

## Endpoint

### POST /game/play/

Makes a move on the Connect Four board using AI inference.

#### Request Format

```json
{
    "board": [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    "current_turn": 1
}
```

**Parameters:**

- `board` (required): A 6x7 array representing the Connect Four board
  - `0` = empty cell
  - `1` = player 1's piece
  - `-1` = player 2's piece
- `current_turn` (required): The current player (`1` or `-1`)

#### Success Response

**Status Code:** 200 OK

```json
{
    "board": [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ],
    "move": 2,
    "row": 5,
    "success": true
}
```

**Response Fields:**

- `board`: Updated board state after the move
- `move`: Column index (0-6) where the piece was placed
- `row`: Row index (0-5) where the piece landed
- `success`: Boolean indicating if the move was successful

#### Error Responses

**Status Code:** 400 Bad Request

Missing required field:
```json
{
    "error": "Missing 'board' in request data"
}
```

Invalid current_turn:
```json
{
    "error": "'current_turn' must be 1 or -1"
}
```

Invalid board size:
```json
{
    "error": "Board must be 6x7, got (2, 7)"
}
```

Board is full:
```json
{
    "error": "No valid moves available (board is full)"
}
```

**Status Code:** 500 Internal Server Error

Model not found:
```json
{
    "error": "Model not found: ..."
}
```

## Usage Examples

### Using curl

```bash
curl -X POST http://localhost:8000/game/play/ \
  -H "Content-Type: application/json" \
  -d '{
    "board": [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0]
    ],
    "current_turn": 1
  }'
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/game/play/"
data = {
    "board": [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    "current_turn": 1
}

response = requests.post(url, json=data)
result = response.json()

print(f"AI chose column: {result['move']}")
print(f"Updated board: {result['board']}")
```

### Using JavaScript fetch

```javascript
const url = 'http://localhost:8000/game/play/';
const data = {
    board: [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    current_turn: 1
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('AI chose column:', result.move);
    console.log('Updated board:', result.board);
})
.catch(error => console.error('Error:', error));
```

## Running the Server

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the Django project directory:
   ```bash
   cd connect4_project
   ```

3. Run migrations (optional, only needed for admin features):
   ```bash
   python manage.py migrate
   ```

4. Start the development server:
   ```bash
   python manage.py runserver
   ```

5. The API will be available at `http://localhost:8000/game/play/`

## Testing

Run the test suite:
```bash
cd connect4_project
python manage.py test game
```

The test suite includes:
- Valid move scenarios (empty board, partially filled board)
- Both player 1 and player -1 turns
- Error handling (missing fields, invalid values, full board)
- Edge cases (invalid board size, empty request body)

