#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// write an simple LRU cache using templates

template <typename K, typename V>
class LRUCache {
  struct Node {
    K key;
    V value;
    Node* prev;
    Node* next;
    Node(K k, V v) : key(k), value(v), prev(nullptr), next(nullptr) {}
  };

  int capacity;
  int size;
  Node* head;
  Node* tail;
  std::unordered_map<K, Node*> map;

  void remove(Node* node)
  {
    if (node->prev) {
      node->prev->next = node->next;
    }
    if (node->next) {
      node->next->prev = node->prev;
    }
    if (node == head) {
      head = node->next;
    }
    if (node == tail) {
      tail = node->prev;
    }
    delete node;
  }

  void add(Node* node)
  {
    if (head == nullptr) {
      head = node;
      tail = node;
    } else {
      node->next = head;
      head->prev = node;
      head = node;
    }
  }

  void moveToHead(Node* node)
  {
    remove(node);
    add(node);
  }

  void removeTail()
  {
    if (tail) {
      map.erase(tail->key);
      remove(tail);
      size--;
    }
  }

 public:
  LRUCache(int capacity)
      : capacity(capacity), size(0), head(nullptr), tail(nullptr)
  {
  }

  ~LRUCache()
  {
    while (head) {
      remove(head);
    }
  }

  V get(K key)
  {
    if (map.find(key) == map.end()) {
      return -1;
    }
    Node* node = map[key];
    moveToHead(node);
    return node->value;
  }

  std::vector<V> getTopK(int k)
  {
    std::vector<V> res;
    Node* node = head;
    while (node && k > 0) {
      res.push_back(node->value);
      node = node->next;
      k--;
    }
    return res;
  }

  void put(K key, V value)
  {
    if (map.find(key) == map.end()) {
      Node* node = new Node(key, value);
      map[key] = node;
      add(node);
      size++;
      if (size > capacity) {
        removeTail();
      }
    } else {
      Node* node = map[key];
      node->value = value;
      moveToHead(node);
    }
  }
};

struct sample_t {
  int a;
  int b;
  sample_t(int a, int b) : a(a), b(b) {}
};

int
main()
{
  std::ofstream out("/mnt/raid0nvme1/xly/swap-engine/test.txt");
  LRUCache<std::string, sample_t> cache(1000);
  for (int i = 0; i < 10000; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // generate uuid string and insert into cache
    cache.put(std::to_string(i), sample_t(i, i));
    auto vec = cache.getTopK(1000);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    out << diff.count() << std::endl;
  }
  out.close();
  return 0;
}