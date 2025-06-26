# 📘 LEOBot Integration Guide for Web Developers

## 🔍 Overview

`LEOBot` is a lightweight, embeddable customer support widget. It shows a contact form and transitions into a chatbot interface after submission. This guide explains how to embed LEOBot on your website using the `load_embedded_leobot(config)` function.

---

## 📦 Requirements

* Your site must allow JavaScript execution.
* You must include a call to `load_embedded_leobot(config)` after the DOM has loaded.

---

## ✅ Quick Start

### 1. **Include the LEOBot Embed Script**

Add this `<script>` tag before the closing `</body>` in your HTML:

```html
<script src="https://leobot.leocdp.com/embed.js"></script>
```

> ⚠️ Replace the `src` with your actual hosting path if self-hosting.

---

### 2. **Initialize LEOBot with Config**

Call `load_embedded_leobot(config)` with your custom settings:

```html
<script>
  document.addEventListener("DOMContentLoaded", function () {
    load_embedded_leobot({
      leobotUrl: "https://leobot.leocdp.com/leobot.html",
      localStorageKey: "leo_bot_state", // optional key for saving state
      text: {
        header: {
          title: "SHOPDUNK XIN CHÀO!",
          subtitle: "Chúng tôi sẵn sàng hỗ trợ bạn!",
        },
        form: {
          title: "Bạn vui lòng điền thông tin để được hỗ trợ:",
          name: "Họ và tên*",
          phone: "Số điện thoại*",
          email: "Địa chỉ email",
          question: "Câu hỏi của bạn*",
          submit: "Gửi thông tin",
        },
        toggleButtonText: "💬 Chat",
      },
    });
  });
</script>
```

---

## ⚙️ Config Options

| Field                   | Type     | Description                                              | Required |
| ----------------------- | -------- | -------------------------------------------------------- | -------- |
| `leobotUrl`             | `string` | Full URL to chatbot iframe                               | ✅        |
| `localStorageKey`       | `string` | Local storage key to persist state (open/form submitted) | ❌        |
| `text.header.title`     | `string` | Header greeting title                                    | ✅        |
| `text.header.subtitle`  | `string` | Header subtitle (supportive message)                     | ✅        |
| `text.form.title`       | `string` | Intro above the contact form                             | ✅        |
| `text.form.name`        | `string` | Placeholder for name input                               | ✅        |
| `text.form.phone`       | `string` | Placeholder for phone input                              | ✅        |
| `text.form.email`       | `string` | Placeholder for email input                              | ✅        |
| `text.form.question`    | `string` | Placeholder for question input                           | ✅        |
| `text.form.submit`      | `string` | Submit button label                                      | ✅        |
| `text.toggleButtonText` | `string` | Floating toggle button text                              | ✅        |

---

## 🧠 How It Works

* A **floating toggle button** (`💬 Chat`) appears on the bottom left.
* When clicked:

  * A **popup form** appears asking for name, phone, and a message.
  * After submission, it switches to the **chatbot iframe**.
* **localStorage** tracks:

  * If the form was submitted (`isSubmitted`)
  * If the popup was last open or closed (`isOpen`)

---

## 🎨 Styling

* All styles are **inline**. No external CSS is required.
* The widget is designed to avoid conflicts with host page styles.

---

## 🛠️ Customization Ideas

* Add `theme: "dark"` or `position: "right"` support in future versions.
* Load different languages by injecting localized config values.
* Connect the form submission to your CRM or backend with a webhook.

---

## 🧪 Debugging

* Open DevTools → Console to see `localStorage.getItem('leo_bot_state')`.
* To reset:

  ```js
  localStorage.removeItem("leo_bot_state");
  ```

---

## ❓Need Help?

Contact the LEOBot team via:

* [Facebook: Trieu Dataism](https://www.facebook.com/dataism.one)
* Email: [trieu@leocdp.com](mailto:trieu@leocdp.com)

