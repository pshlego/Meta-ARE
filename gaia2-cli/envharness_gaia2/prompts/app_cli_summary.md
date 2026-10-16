# Gaia2 app CLIs available to the agent (via OpenClaw `exec`)

Each app is a shell command; subcommands below (state-changing ones marked [write] when the name implies it).

## `calendar`
- `calendar add-event` [write] — Add a calendar event to the calendar.
- `calendar delete-event` [write] — Delete a calendar event from the calendar.
- `calendar get-all-tags` — Get all tags from the calendar.
- `calendar get-event` — Read a calendar event from the calendar.
- `calendar get-events` — Get calendar events that have any time overlap with the specified date range (excludes...
- `calendar get-events-by-tag` — Get calendar events from the calendar by tag.
- `calendar search-events` — Searches for calendar events based on a query string.
- `calendar today-events` — Read today's calendar events from the calendar.

## `contacts`
- `contacts add-new-contact` [write] — Adds a new contact to the contacts app.
- `contacts delete-contact` [write] — Deletes a specific contact by contact_id.
- `contacts edit-contact` [write] — Edits specific fields of a contact by contact_id.
- `contacts get-contact` — Gets a specific contact by contact_id.
- `contacts get-contacts` — Gets the list of contacts starting from a specified offset.
- `contacts get-current-user-details` — Gets the current user's details including name, phone number, email address,...
- `contacts search-contacts` — Searches for contacts based on a query string.

## `emails`
- `emails delete-email` [write] — Delete an email (moves it to TRASH).
- `emails download-attachments` — Download attachments from an email to the local filesystem.
- `emails forward-email` [write] — Forward an email to new recipients.
- `emails get-email-by-id` — Get an email by its ID, marking it as read.
- `emails get-email-by-index` — Get an email by its index in a folder, marking it as read.
- `emails list-emails` — List emails in a folder with pagination.
- `emails move-email` [write] — Move an email from one folder to another.
- `emails reply-to-email` [write] — Reply to an email.
- `emails search-emails` — Search emails by query string across sender, recipients, subject, and content.
- `emails send-email` [write] — Send an email to the specified recipients.

## `messages`
- `messages add-participant-to-conversation` [write] — Add a user to a conversation.
- `messages change-conversation-title` [write] — Change the title of a conversation.
- `messages create-group-conversation` [write] — Create a new group conversation.
- `messages download-attachment` — Download an attachment from a message.
- `messages get-existing-conversation-ids` — Get conversation ids matching the exact set of participants.
- `messages get-user-id` — Get user id for a given user name (exact match).
- `messages get-user-name-from-id` — Get user name for a given user id (exact match).
- `messages list-conversations-by-participant` — List conversations that include the given participant.
- `messages list-recent-conversations` — List conversations ordered by most recent update.
- `messages lookup-user-id` — Fuzzy lookup of user id by name.
- `messages read-conversation` — Read messages from a conversation (most recent first).
- `messages regex-search` — Search conversations using a regex pattern (case-insensitive).
- `messages remove-participant-from-conversation` [write] — Remove a user from a conversation.
- `messages search` — Search conversations by participants, title, and message content.
- `messages send-message` [write] — Send a message to a user.
- `messages send-message-to-group-conversation` [write] — Send a message to an existing group conversation (3+ participants).

## `chats`
- `chats add-participant-to-conversation` [write] — Add a user to a conversation.
- `chats change-conversation-title` [write] — Change the title of a conversation.
- `chats create-group-conversation` [write] — Create a new group conversation.
- `chats download-attachment` — Download an attachment from a message.
- `chats get-existing-conversation-ids` — Get conversation ids matching the exact set of participants.
- `chats get-user-id` — Get user id for a given user name (exact match).
- `chats get-user-name-from-id` — Get user name for a given user id (exact match).
- `chats list-conversations-by-participant` — List conversations that include the given participant.
- `chats list-recent-conversations` — List conversations ordered by most recent update.
- `chats lookup-user-id` — Fuzzy lookup of user id by name.
- `chats read-conversation` — Read messages from a conversation (most recent first).
- `chats regex-search` — Search conversations using a regex pattern (case-insensitive).
- `chats remove-participant-from-conversation` [write] — Remove a user from a conversation.
- `chats search` — Search conversations by participants, title, and message content.
- `chats send-message` [write] — Send a message to a user.
- `chats send-message-to-group-conversation` [write] — Send a message to an existing group conversation (3+ participants).

## `rent-a-flat`
- `rent-a-flat get-apartment-details` — Get apartment details for a given apartment id.
- `rent-a-flat list-all-apartments` — List all apartments in the catalog.
- `rent-a-flat list-saved-apartments` [write] — List apartments saved to favorites.
- `rent-a-flat remove-saved-apartment` [write] — Remove an apartment from favorites.
- `rent-a-flat save-apartment` [write] — Save an apartment to favorites.
- `rent-a-flat search-apartments` — Search for apartments based on multiple filters.

## `city`
- `city get-api-call-count` — Get the current API call count.
- `city get-api-call-limit` — Get the API call limit for the service.
- `city get-crime-rate` — Get crime rate for a given zip code (rate-limited).

## `cabs`
- `cabs get-current-ride-status` — Check the status of the current ride.
- `cabs get-quotation` — Calculate price and estimated delay for a ride.
- `cabs get-ride` — Get a specific ride from ride history by index.
- `cabs get-ride-history` — Get ride history with pagination.
- `cabs get-ride-history-length` — Get the total number of rides in history.
- `cabs list-rides` — List all rides available between two locations.
- `cabs order-ride` [write] — Order a ride and return ride details.
- `cabs user-cancel-ride` [write] — Cancel the current ride.

## `shopping`
- `shopping add-to-cart` [write] — Add an item to the cart.
- `shopping cancel-order` [write] — Cancel an order.
- `shopping checkout` [write] — Checkout the current cart and create an order.
- `shopping get-all-discount-codes` — List all available discount codes.
- `shopping get-discount-code-info` — Get information about a specific discount code.
- `shopping get-order-details` [write] — Get details for a specific order.
- `shopping get-product-details` — Get product details for a given product ID.
- `shopping list-all-products` — List all products in the catalog.
- `shopping list-cart` — List the contents of the cart.
- `shopping list-orders` [write] — List all orders.
- `shopping remove-from-cart` [write] — Remove an item from the cart.
- `shopping search-products` — Search for products by name (case-insensitive substring match).

## `cloud-drive`
- `cloud-drive cat` — Read the contents of a file and print to stdout.
- `cloud-drive exists` — Check if a file or directory exists.
- `cloud-drive info` — Get metadata about a file or directory.
- `cloud-drive ls` — List the contents of a directory.
- `cloud-drive makedirs` — Create nested directories (like mkdir -p).
- `cloud-drive mkdir` — Create a directory.
- `cloud-drive mv` — Move a file or directory.
- `cloud-drive rm` — Remove a file or directory.
- `cloud-drive rmdir` — Remove an empty directory.
- `cloud-drive tree` — Generate a directory tree listing.
