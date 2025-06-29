// Run `go mod init redis-loader`
// Run `go get github.com/redis/go-redis/v9`
// Run `go get golang.org/x/net/context`
// Run `go run keywordscache.go`

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/redis/go-redis/v9"
	"golang.org/x/net/context"
)

func main() {
	ctx := context.Background()

	// Connect to Redis using default DB 0
	rdb := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0, // Use default DB
	})

	// Open the keyword file
	file, err := os.Open("../../data/resume_keywords_db.txt")
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		keyword := strings.TrimSpace(scanner.Text())
		if keyword != "" {
			// Use namespace prefix
			key := fmt.Sprintf("keyword:%s", keyword)
			err := rdb.Set(ctx, key, 1, 0).Err() // Set value with no expiration
			if err != nil {
				log.Printf("Failed to set key %s: %v", key, err)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Scanner error: %v", err)
	}

	fmt.Println("Keywords loaded into Redis with 'keyword:' namespace")
}
