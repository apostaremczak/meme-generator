import { BaseTokenizer } from 'tokenizers';

async function app() {
  console.log('Loading tokenizer..');

  const wordPieceTokenizer = await BaseTokenizer.fromOptions({ vocabFile: "./vocab.json" });
  console.log('Successfully loaded tokenizer');
}

app();
