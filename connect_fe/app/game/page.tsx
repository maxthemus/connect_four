'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

type Board = number[][];
type GameState = 'player_turn' | 'ai_thinking' | 'player_won' | 'ai_won' | 'draw';

const ROWS = 6;
const COLS = 7;
const PLAYER = 1;
const AI = -1;

export default function GamePage() {
  const [board, setBoard] = useState<Board>(createEmptyBoard());
  const [gameState, setGameState] = useState<GameState>('player_turn');
  const [hoveredCol, setHoveredCol] = useState<number | null>(null);
  const [lastMove, setLastMove] = useState<{ row: number; col: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  function createEmptyBoard(): Board {
    return Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
  }

  function checkWinner(board: Board, player: number): boolean {
    // Check horizontal
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS - 3; col++) {
        if (
          board[row][col] === player &&
          board[row][col + 1] === player &&
          board[row][col + 2] === player &&
          board[row][col + 3] === player
        ) {
          return true;
        }
      }
    }

    // Check vertical
    for (let row = 0; row < ROWS - 3; row++) {
      for (let col = 0; col < COLS; col++) {
        if (
          board[row][col] === player &&
          board[row + 1][col] === player &&
          board[row + 2][col] === player &&
          board[row + 3][col] === player
        ) {
          return true;
        }
      }
    }

    // Check diagonal (down-right)
    for (let row = 0; row < ROWS - 3; row++) {
      for (let col = 0; col < COLS - 3; col++) {
        if (
          board[row][col] === player &&
          board[row + 1][col + 1] === player &&
          board[row + 2][col + 2] === player &&
          board[row + 3][col + 3] === player
        ) {
          return true;
        }
      }
    }

    // Check diagonal (down-left)
    for (let row = 0; row < ROWS - 3; row++) {
      for (let col = 3; col < COLS; col++) {
        if (
          board[row][col] === player &&
          board[row + 1][col - 1] === player &&
          board[row + 2][col - 2] === player &&
          board[row + 3][col - 3] === player
        ) {
          return true;
        }
      }
    }

    return false;
  }

  function isBoardFull(board: Board): boolean {
    return board[0].every(cell => cell !== 0);
  }

  function isColumnFull(col: number): boolean {
    return board[0][col] !== 0;
  }

  async function makeAIMove(currentBoard: Board) {
    setGameState('ai_thinking');
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/game/play/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          board: currentBoard,
          current_turn: AI,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get AI move');
      }

      const data = await response.json();
      const updatedBoard = data.board;
      
      setBoard(updatedBoard);
      setLastMove({ row: data.row, col: data.move });

      // Check if AI won
      if (checkWinner(updatedBoard, AI)) {
        setGameState('ai_won');
      } else if (isBoardFull(updatedBoard)) {
        setGameState('draw');
      } else {
        setGameState('player_turn');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to API');
      setGameState('player_turn');
    }
  }

  function handleColumnClick(col: number) {
    if (gameState !== 'player_turn' || isColumnFull(col)) {
      return;
    }

    // Find the lowest empty row in the column
    let row = -1;
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r][col] === 0) {
        row = r;
        break;
      }
    }

    if (row === -1) return;

    // Apply player's move
    const newBoard = board.map(r => [...r]);
    newBoard[row][col] = PLAYER;
    setBoard(newBoard);
    setLastMove({ row, col });

    // Check if player won
    if (checkWinner(newBoard, PLAYER)) {
      setGameState('player_won');
      return;
    }

    // Check for draw
    if (isBoardFull(newBoard)) {
      setGameState('draw');
      return;
    }

    // Let AI make its move
    makeAIMove(newBoard);
  }

  function resetGame() {
    setBoard(createEmptyBoard());
    setGameState('player_turn');
    setLastMove(null);
    setError(null);
  }

  function getCellColor(value: number, row: number, col: number): string {
    const isLastMove = lastMove?.row === row && lastMove?.col === col;
    
    if (value === PLAYER) {
      return isLastMove ? 'bg-red-500 ring-4 ring-red-300' : 'bg-red-500';
    } else if (value === AI) {
      return isLastMove ? 'bg-yellow-400 ring-4 ring-yellow-200' : 'bg-yellow-400';
    }
    return 'bg-white';
  }

  function getGameStateMessage(): { text: string; color: string } {
    switch (gameState) {
      case 'player_turn':
        return { text: 'Your Turn - Click a column to drop your piece', color: 'text-blue-600' };
      case 'ai_thinking':
        return { text: 'AI is thinking...', color: 'text-purple-600 animate-pulse' };
      case 'player_won':
        return { text: '🎉 You Won!', color: 'text-green-600' };
      case 'ai_won':
        return { text: 'AI Won! Better luck next time.', color: 'text-red-600' };
      case 'draw':
        return { text: "It's a Draw!", color: 'text-gray-600' };
      default:
        return { text: '', color: '' };
    }
  }

  const stateMessage = getGameStateMessage();
  const isGameOver = ['player_won', 'ai_won', 'draw'].includes(gameState);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 p-4">
      {/* Game Over Modal */}
      {isGameOver && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full animate-in fade-in zoom-in duration-300">
            <div className="text-center">
              {/* Icon/Emoji based on result */}
              <div className="text-6xl mb-4">
                {gameState === 'player_won' && '🎉'}
                {gameState === 'ai_won' && '🤖'}
                {gameState === 'draw' && '🤝'}
              </div>
              
              {/* Winner message */}
              <h2 className="text-3xl font-bold mb-2">
                {gameState === 'player_won' && 'You Won!'}
                {gameState === 'ai_won' && 'AI Wins!'}
                {gameState === 'draw' && "It's a Draw!"}
              </h2>
              
              <p className="text-gray-600 mb-6">
                {gameState === 'player_won' && 'Congratulations! You beat the AI!'}
                {gameState === 'ai_won' && 'Better luck next time!'}
                {gameState === 'draw' && 'The board is full with no winner!'}
              </p>
              
              {/* Buttons */}
              <div className="flex flex-col gap-3">
                <button
                  onClick={resetGame}
                  className="w-full py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-bold text-lg hover:scale-105 transition-all shadow-lg"
                >
                  Play Again
                </button>
                <Link
                  href="/"
                  className="w-full py-4 bg-gray-200 text-gray-700 rounded-xl font-semibold text-lg hover:bg-gray-300 transition-all text-center"
                >
                  Back to Menu
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <Link
            href="/"
            className="px-3 py-2 bg-white/20 backdrop-blur-sm rounded-lg text-white text-sm font-semibold hover:bg-white/30 transition-all"
          >
            ← Back
          </Link>
          <h1 className="text-2xl font-bold text-white drop-shadow-lg">
            Connect Four
          </h1>
          <button
            onClick={resetGame}
            className="px-3 py-2 bg-white/20 backdrop-blur-sm rounded-lg text-white text-sm font-semibold hover:bg-white/30 transition-all"
          >
            New Game
          </button>
        </div>

        {/* Game State Message */}
        <div className="text-center mb-4">
          <p className={`text-lg font-bold ${stateMessage.color} bg-white/90 rounded-lg py-3 px-4 shadow-lg`}>
            {stateMessage.text}
          </p>
          {error && (
            <p className="text-sm text-red-600 bg-white/90 rounded-lg py-2 px-3 mt-2 shadow-lg">
              ⚠️ {error}
            </p>
          )}
        </div>

        {/* Game Board */}
        <div className="bg-blue-700 rounded-2xl p-4 shadow-2xl max-w-xl mx-auto">
          <div className="grid grid-cols-7 gap-2">
            {Array.from({ length: COLS }, (_, col) => (
              <div key={col} className="flex flex-col gap-2">
                {/* Column indicator on hover */}
                <div
                  className={`h-2 rounded-full transition-all ${
                    hoveredCol === col && gameState === 'player_turn' && !isColumnFull(col)
                      ? 'bg-red-400 opacity-100'
                      : 'bg-transparent opacity-0'
                  }`}
                />
                {Array.from({ length: ROWS }, (_, row) => (
                  <div
                    key={`${row}-${col}`}
                    className={`aspect-square rounded-full border-2 border-blue-800 shadow-inner transition-all cursor-pointer ${getCellColor(
                      board[row][col],
                      row,
                      col
                    )} ${
                      gameState === 'player_turn' && !isColumnFull(col)
                        ? 'hover:scale-105'
                        : ''
                    }`}
                    onClick={() => handleColumnClick(col)}
                    onMouseEnter={() => setHoveredCol(col)}
                    onMouseLeave={() => setHoveredCol(null)}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 flex justify-center gap-6">
          <div className="flex items-center gap-2 bg-white/90 rounded-lg px-3 py-2 shadow-lg">
            <div className="w-5 h-5 bg-red-500 rounded-full" />
            <span className="text-sm font-semibold text-gray-700">You</span>
          </div>
          <div className="flex items-center gap-2 bg-white/90 rounded-lg px-3 py-2 shadow-lg">
            <div className="w-5 h-5 bg-yellow-400 rounded-full" />
            <span className="text-sm font-semibold text-gray-700">AI</span>
          </div>
        </div>

        {/* Game Over Actions - Removed since we have modal now */}
      </div>
    </div>
  );
}
