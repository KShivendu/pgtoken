// Helper for paper/compress_by_length.py: per-value FSST, and the Schmidt et al. 16-bit token
// table, on many strings at once. Python can't call either at native speed, so the script writes
// the strings to a file, runs this, and reads one record per string back.
//
//   fsst_cli fsst    train.bin test.bin out.bin
//   fsst_cli schmidt vocab.bin test.bin out.bin
//
// train.bin / test.bin: repeated [u32 length][bytes], one record per value.
// vocab.bin:            repeated [u8 length][bytes], one record per token (ids = file order).
//
// fsst: one symbol table is trained on all train strings (fsst_create samples them itself), then
//   every test string is compressed on its own and decompressed back, best of 3 timed runs, and
//   checked byte for byte. out.bin = [u32 table_bytes][u64 train_ns] then per string
//   [u32 compressed_bytes][u32 decompress_ns].
// schmidt: greedy longest match over the vocabulary, tokens of 1..7 bytes, as their
//   TokenizerCompressor does (2 bytes per token, a single byte is always a token). out.bin = per
//   string [u32 tokens][u32 0].
//
// Build (compress_by_length.py does this on first use):
//   g++ -O3 -DNDEBUG -std=c++17 -c fsst/libfsst.cpp ; g++ -O1 -std=c++17 -c fsst/fsst_avx512.cpp
//   g++ -O3 -std=c++17 -Ifsst fsst_cli.cpp libfsst.o fsst_avx512.o -o fsst_cli -lpthread

#include "fsst.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

static std::vector<std::string> read_values(const char* path) {
   std::ifstream f(path, std::ios::binary);
   if (!f) throw std::runtime_error(std::string("cannot open ") + path);
   std::vector<std::string> out;
   uint32_t n;
   while (f.read(reinterpret_cast<char*>(&n), 4)) {
      std::string s(n, '\0');
      if (n && !f.read(s.data(), n)) throw std::runtime_error("truncated record");
      out.push_back(std::move(s));
   }
   return out;
}

static uint64_t now_ns() {
   return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static int run_fsst(const char* train_p, const char* test_p, const char* out_p) {
   auto train = read_values(train_p);
   auto test = read_values(test_p);
   std::vector<size_t> tlen;
   std::vector<const unsigned char*> tptr;
   for (auto& s : train) {
      tlen.push_back(s.size());
      tptr.push_back(reinterpret_cast<const unsigned char*>(s.data()));
   }
   uint64_t t0 = now_ns();
   fsst_encoder_t* enc = fsst_create(train.size(), tlen.data(), tptr.data(), 0);
   uint64_t train_ns = now_ns() - t0;
   std::vector<unsigned char> hdr(FSST_MAXHEADER);
   uint32_t table_bytes = fsst_export(enc, hdr.data());
   fsst_decoder_t dec = fsst_decoder(enc);

   std::ofstream out(out_p, std::ios::binary);
   out.write(reinterpret_cast<const char*>(&table_bytes), 4);
   out.write(reinterpret_cast<const char*>(&train_ns), 8);
   std::vector<unsigned char> cbuf, dbuf;
   for (auto& s : test) {
      size_t len = s.size();
      const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
      cbuf.resize(16 + 2 * len);
      size_t clen = 0;
      unsigned char* cptr = nullptr;
      if (fsst_compress(enc, 1, &len, &p, cbuf.size(), cbuf.data(), &clen, &cptr) != 1)
         throw std::runtime_error("fsst_compress: output buffer too small");
      dbuf.resize(len + 64);
      uint64_t best = UINT64_MAX;
      size_t dlen = 0;
      for (int r = 0; r < 3; r++) {
         uint64_t a = now_ns();
         dlen = fsst_decompress(&dec, clen, cptr, dbuf.size(), dbuf.data());
         uint64_t b = now_ns() - a;
         if (b < best) best = b;
      }
      if (dlen != len || std::memcmp(dbuf.data(), s.data(), len) != 0)
         throw std::runtime_error("fsst round trip mismatch");
      uint32_t c32 = static_cast<uint32_t>(clen), t32 = static_cast<uint32_t>(best);
      out.write(reinterpret_cast<const char*>(&c32), 4);
      out.write(reinterpret_cast<const char*>(&t32), 4);
   }
   fsst_destroy(enc);
   return 0;
}

static int run_schmidt(const char* vocab_p, const char* test_p, const char* out_p) {
   std::ifstream f(vocab_p, std::ios::binary);
   if (!f) throw std::runtime_error("cannot open vocab");
   std::unordered_set<std::string> by_len[8];
   unsigned char n;
   size_t count = 0;
   while (f.read(reinterpret_cast<char*>(&n), 1)) {
      std::string s(n, '\0');
      f.read(s.data(), n);
      if (n == 0 || n > 7) throw std::runtime_error("token length outside 1..7");
      by_len[n].insert(s);
      count++;
   }
   if (count > 65536) throw std::runtime_error("more than 65536 tokens");
   auto test = read_values(test_p);
   std::ofstream out(out_p, std::ios::binary);
   std::string key;
   for (auto& s : test) {
      uint32_t tokens = 0;
      size_t i = 0;
      while (i < s.size()) {
         size_t take = 1;  // every single byte is a token
         for (size_t L = std::min<size_t>(7, s.size() - i); L >= 2; L--) {
            key.assign(s, i, L);
            if (by_len[L].count(key)) {
               take = L;
               break;
            }
         }
         i += take;
         tokens++;
      }
      uint32_t zero = 0;
      out.write(reinterpret_cast<const char*>(&tokens), 4);
      out.write(reinterpret_cast<const char*>(&zero), 4);
   }
   return 0;
}

int main(int argc, char** argv) {
   if (argc != 5) {
      std::cerr << "usage: fsst_cli fsst|schmidt <train.bin|vocab.bin> test.bin out.bin\n";
      return 2;
   }
   try {
      if (std::string(argv[1]) == "fsst") return run_fsst(argv[2], argv[3], argv[4]);
      if (std::string(argv[1]) == "schmidt") return run_schmidt(argv[2], argv[3], argv[4]);
   } catch (const std::exception& e) {
      std::cerr << "fsst_cli: " << e.what() << "\n";
      return 1;
   }
   std::cerr << "unknown mode\n";
   return 2;
}
