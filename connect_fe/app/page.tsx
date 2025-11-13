import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600 font-sans">
      <main className="flex flex-col items-center justify-center gap-8 p-8">
        <div className="text-center">
          <h1 className="text-6xl font-bold text-white mb-4 drop-shadow-lg">
            Connect Four
          </h1>
          <p className="text-xl text-white/90 mb-8">
            Challenge the AI or play with friends
          </p>
        </div>
        
        <nav className="flex flex-col gap-4 w-full max-w-md">
          <Link
            href="/game"
            className="flex h-16 items-center justify-center rounded-xl bg-white px-8 text-xl font-semibold text-blue-600 transition-all hover:scale-105 hover:shadow-2xl"
          >
            Play Game
          </Link>
          
          <button
            className="flex h-16 items-center justify-center rounded-xl bg-white/20 px-8 text-xl font-semibold text-white backdrop-blur-sm transition-all hover:scale-105 hover:bg-white/30"
            disabled
          >
            How to Play
          </button>
          
          <button
            className="flex h-16 items-center justify-center rounded-xl bg-white/20 px-8 text-xl font-semibold text-white backdrop-blur-sm transition-all hover:scale-105 hover:bg-white/30"
            disabled
          >
            Settings
          </button>
        </nav>
      </main>
    </div>
  );
}
